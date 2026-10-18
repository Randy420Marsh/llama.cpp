"""
Self-signed TLS certificate for local and LAN use.

Why the app needs HTTPS at all: browsers only expose the microphone on a
secure origin. Over plain http the mic works on localhost and nowhere else,
so voice-to-text is dead on a phone the moment you leave the desk. Serving
https fixes that, and stops the LAN traffic being readable.

Why a certificate is generated rather than typed: modern browsers ignore the
Common Name entirely and match the host against subjectAltName. A certificate
without a SAN for the exact name in the address bar is rejected outright, and
"the exact name" is different for every device -- 127.0.0.1 from this PC,
192.168.x.x from a phone, the hostname from a laptop. So every local address
is enumerated at generation time and all of them go in.

    python certs_util.py            create or renew certs/server.crt
    python certs_util.py --show     print what the current certificate covers
    python certs_util.py --force    regenerate even if the current one is fine
"""
from __future__ import annotations

import datetime
import ipaddress
import os
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CERT_DIR = ROOT / "certs"
CERT_FILE = CERT_DIR / "server.crt"
KEY_FILE = CERT_DIR / "server.key"

# Safari refuses server certificates valid for more than 398 days, so a
# 10-year certificate would work everywhere except an iPhone. Staying under
# the limit and renewing automatically is the only option that works on every
# device without anyone having to remember anything.
DAYS = 397
RENEW_WITHIN_DAYS = 30


def local_addresses() -> tuple[list[str], list[str]]:
    """Every name and IP this machine can plausibly be reached by."""
    names = ["localhost"]
    ips = ["127.0.0.1", "::1"]

    host = socket.gethostname()
    for n in (host, host.lower(), host + ".local"):
        if n not in names:
            names.append(n)

    # Every address on every interface, both families.
    try:
        for fam, _, _, _, sa in socket.getaddrinfo(host, None):
            ip = sa[0].split("%")[0]           # strip the zone id on link-local
            if ip not in ips:
                ips.append(ip)
    except OSError:
        pass

    # The address that actually carries the default route, in case the
    # hostname lookup above missed it.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 80))          # TEST-NET-1: reserved, unroutable
        ip = s.getsockname()[0]
        if ip not in ips:
            ips.append(ip)
    except OSError:
        pass
    finally:
        s.close()

    return names, ips


def _valid(cert_path: Path, key_path: Path) -> tuple[bool, str]:
    """Is the certificate on disk usable, and does it still cover us?"""
    if not cert_path.exists() or not key_path.exists():
        return False, "no certificate yet"
    try:
        from cryptography import x509
    except ImportError:
        return False, "cryptography not installed"
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    except Exception as e:
        return False, f"unreadable ({e})"

    now = datetime.datetime.now(datetime.timezone.utc)
    left = cert.not_valid_after_utc - now
    if left.days < RENEW_WITHIN_DAYS:
        return False, f"expires in {left.days} days"

    try:
        san = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName).value
        have_ips = {str(x) for x in san.get_values_for_type(x509.IPAddress)}
        have_names = {str(x) for x in san.get_values_for_type(x509.DNSName)}
    except x509.ExtensionNotFound:
        return False, "certificate has no subjectAltName"

    names, ips = local_addresses()
    missing = [x for x in ips if x not in have_ips] + \
              [x for x in names if x not in have_names]
    if missing:
        # The LAN address changes with DHCP; a certificate that no longer
        # names it fails on exactly the device it was made for.
        return False, "does not cover " + ", ".join(missing[:4])
    return True, f"valid for {left.days} more days"


def ensure_cert(force: bool = False) -> tuple[Path, Path, str]:
    """Return (cert, key, note), generating a new pair only when needed."""
    ok, why = _valid(CERT_FILE, KEY_FILE)
    if ok and not force:
        return CERT_FILE, KEY_FILE, why

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    names, ips = local_addresses()
    alt: list[x509.GeneralName] = [x509.DNSName(n) for n in names]
    for ip in ips:
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass

    # RSA-2048 rather than an elliptic curve: it is what every phone, smart TV
    # and elderly browser on a home network accepts without argument.
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, socket.gethostname()),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "llama.cpp-server-tts"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)                       # self-signed
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))   # clock skew
        .not_valid_after(now + datetime.timedelta(days=DAYS))
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None),
                       critical=True)
        .add_extension(x509.ExtendedKeyUsage([
            x509.ObjectIdentifier("1.3.6.1.5.5.7.3.1")]),         # serverAuth
            critical=False)
        .sign(key, hashes.SHA256())
    )

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    KEY_FILE.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()))
    # The key is unencrypted, so on a multi-user Linux box it must not be
    # world-readable. Windows ignores the mode; there the NTFS ACL applies.
    try:
        os.chmod(KEY_FILE, 0o600)
    except OSError:
        pass
    return CERT_FILE, KEY_FILE, f"generated ({why})"


def describe() -> str:
    from cryptography import x509
    if not CERT_FILE.exists():
        return "no certificate at " + str(CERT_FILE)
    cert = x509.load_pem_x509_certificate(CERT_FILE.read_bytes())
    san = cert.extensions.get_extension_for_class(
        x509.SubjectAlternativeName).value
    covers = [str(x) for x in san.get_values_for_type(x509.DNSName)] + \
             [str(x) for x in san.get_values_for_type(x509.IPAddress)]
    ok, why = _valid(CERT_FILE, KEY_FILE)
    return (f"{CERT_FILE}\n  expires {cert.not_valid_after_utc:%Y-%m-%d} "
            f"({why})\n  covers  " + ", ".join(covers))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.show:
        print(describe())
    else:
        c, k, note = ensure_cert(force=a.force)
        print(f"[OK] {note}")
        print(describe())
