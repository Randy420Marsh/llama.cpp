/**
 * Speech: what the speech server offers and the read-aloud player.
 *
 * The lists (engines, their voices and models) come from the server, so the
 * Voice settings and the read-aloud button only ever offer what can really be
 * spoken. Reading is sentence by sentence: the next piece is synthesised while
 * the current one plays, so a long reply starts within a second and never
 * waits for the whole answer to be rendered.
 */
import { conversationsStore } from './conversations/index.svelte';
import { settingsStore } from './settings/index.svelte';
import { browser } from '$app/environment';
import { SETTINGS_KEYS } from '$lib/constants';
import {
	getSpeechEngines,
	getSpeechStatus,
	pickEngine,
	resolveSpeechUrl,
	type SpeechEngine,
	type SpeechStatus,
	splitForSpeech,
	type SynthesisRequest,
	type SynthesisResult,
	synthesize,
	toSpeechText
} from '$lib/utils/tts';
import { toast } from 'svelte-sonner';

class SpeechStore {
	checkedUrl = $state('');
	checking = $state(false);
	engines = $state<SpeechEngine[]>([]);
	error = $state('');
	/** waiting for the first piece of audio */
	loading = $state(false);

	/** id of the message being read, null when silent */
	speakingId = $state<string | null>(null);
	status = $state<SpeechStatus | null>(null);

	private abort: AbortController | null = null;
	private audio: HTMLAudioElement | null = null;
	private lastCheck = 0;
	private run = 0;

	/** the server answered at the configured URL */
	get available(): boolean {
		return this.status !== null && this.checkedUrl === this.url;
	}

	get url(): string {
		return resolveSpeechUrl(settingsStore.config[SETTINGS_KEYS.TTS_SERVER_URL]);
	}

	/** Called when a reply has finished streaming. */
	autoRead(messageId: string, content: string, convId: string): void {
		if (!settingsStore.config[SETTINGS_KEYS.TTS_AUTO_READ]) return;

		if (convId !== conversationsStore.activeConversation?.id) return;

		if (!content.trim()) return;

		void this.speak(messageId, content);
	}

	/** re-check when the URL changed or the last answer is older than maxAgeMs */
	ensureFresh(maxAgeMs = 60_000): void {
		if (!browser || this.checking) return;

		if (this.checkedUrl !== this.url || Date.now() - this.lastCheck > maxAgeMs) {
			void this.refresh();
		}
	}

	async refresh(url = this.url): Promise<boolean> {
		if (!browser) return false;

		this.checking = true;

		try {
			const [status, engines] = await Promise.all([getSpeechStatus(url), getSpeechEngines(url)]);

			this.status = status;
			this.engines = engines;
			this.error = '';
		} catch (error) {
			this.status = null;
			this.engines = [];
			this.error = error instanceof Error ? error.message : String(error);
		} finally {
			this.checkedUrl = url;
			this.lastCheck = Date.now();
			this.checking = false;
		}

		return this.status !== null;
	}

	/** engine, voice, model, language and speed from the saved settings */
	request(config: Record<string, unknown> = settingsStore.config): SynthesisRequest {
		const engine = this.resolveEngine(String(config[SETTINGS_KEYS.TTS_ENGINE] ?? ''));
		const voice = String(config[SETTINGS_KEYS.TTS_VOICE] ?? '');
		const model = String(config[SETTINGS_KEYS.TTS_MODEL] ?? '');

		// a voice or model saved for another engine is not this engine's: use its default
		return {
			engine: engine?.name,
			language: String(config[SETTINGS_KEYS.TTS_LANGUAGE] ?? 'auto'),
			model: engine?.models.some((m) => m.id === model) ? model : '',
			speed: Number(config[SETTINGS_KEYS.TTS_SPEED] ?? 1) || 1,
			voice: engine?.voices.some((v) => v.id === voice) ? voice : ''
		};
	}

	/** the engine a request will use: the configured one, else the best installed */
	resolveEngine(name?: string): SpeechEngine | undefined {
		return pickEngine(
			this.engines,
			name ?? String(settingsStore.config[SETTINGS_KEYS.TTS_ENGINE] ?? '')
		);
	}

	/**
	 * Read `markdown` aloud as message `id`. `req` and `serverUrl` override the
	 * saved settings (the Voice settings test button plays unsaved choices).
	 */
	async speak(
		id: string,
		markdown: string,
		req?: SynthesisRequest,
		serverUrl?: string
	): Promise<void> {
		this.stop();
		const run = ++this.run;

		if (!serverUrl) {
			if (!this.available) {
				await this.refresh();
			}

			if (run !== this.run) return;

			if (!this.available) {
				toast.error(`Read aloud: no speech server at ${this.url} (Settings → Voice)`);

				return;
			}
		}

		const pieces = splitForSpeech(toSpeechText(markdown));

		if (pieces.length === 0) return;

		const r = req ?? this.request();
		const url = serverUrl ?? this.url;
		const abort = new AbortController();

		this.abort = abort;
		this.speakingId = id;
		this.loading = true;

		const fetchPiece = (i: number): Promise<SynthesisResult> => {
			const p = synthesize(url, pieces[i], r, abort.signal);

			p.catch(() => {}); // a stop() aborts pieces nobody will await

			return p;
		};

		let next: Promise<SynthesisResult> | null = fetchPiece(0);
		let warned = false;

		try {
			for (let i = 0; i < pieces.length && next; i++) {
				const res: SynthesisResult = await next;

				if (run !== this.run) return;

				// fetch the following piece while this one plays
				next = i + 1 < pieces.length ? fetchPiece(i + 1) : null;

				if (res.fallback && !warned) {
					warned = true;
					toast.warning(`${r.engine ?? 'TTS'} failed, read with ${res.fallback}: ${res.error}`);
				}

				this.loading = false;
				await this.play(res, r.speed ?? 1, run);

				if (run !== this.run) return;
			}
		} catch (error) {
			if (run === this.run && !abort.signal.aborted) {
				toast.error(`Read aloud failed: ${error instanceof Error ? error.message : String(error)}`);
			}
		} finally {
			if (run === this.run) {
				this.speakingId = null;
				this.loading = false;
				this.abort = null;
			}
		}
	}

	stop(): void {
		this.run++;
		this.abort?.abort();
		this.abort = null;

		if (this.audio) {
			this.audio.pause();
			this.audio = null;
		}

		this.speakingId = null;
		this.loading = false;
	}

	/** Read `markdown` aloud as message `id`; a second call for the same id stops it. */
	async toggle(id: string, markdown: string): Promise<void> {
		if (this.speakingId === id) {
			this.stop();

			return;
		}

		await this.speak(id, markdown);
	}

	private play(res: SynthesisResult, speed: number, run: number): Promise<void> {
		return new Promise((resolve, reject) => {
			const src = URL.createObjectURL(res.blob);
			const audio = new Audio(src);
			const finish = () => {
				URL.revokeObjectURL(src);

				if (this.audio === audio) this.audio = null;

				resolve();
			};

			this.audio = audio;
			audio.preservesPitch = true;
			// engines that cannot change their rate are sped up by the player instead
			audio.playbackRate = res.speedNative ? 1 : speed;
			audio.onended = finish;
			audio.onpause = () => {
				if (run !== this.run) finish();
			};
			audio.onerror = () => {
				URL.revokeObjectURL(src);
				reject(new Error('the browser could not play the audio'));
			};
			audio.play().catch(reject);
		});
	}
}

export const speechStore = new SpeechStore();
