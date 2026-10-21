/**
 * Prepare an in-DOM serialized svg (svg.outerHTML) for download as a
 * standalone image/svg+xml file. The DOM serializes HTML void tags like
 * <br> unclosed inside foreignObject, which is invalid XML: self-close
 * every void tag and prepend the XML declaration when it is absent, so
 * the file parses in strict XML viewers (Firefox, Chrome, Inkscape).
 */

const VOID_TAG_RE =
	/<(area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr)(\s[^<>]*)?\/?>/gi;

const XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8"?>\n';

export function toStandaloneSvg(svgHtml: string): string {
	const trimmed = svgHtml.trim();
	if (!trimmed) return '';

	const selfClosed = trimmed.replace(VOID_TAG_RE, (match) =>
		match.endsWith('/>') ? match : match.slice(0, -1) + '/>'
	);

	if (selfClosed.startsWith('<?xml')) return selfClosed;
	return XML_DECLARATION + selfClosed;
}
