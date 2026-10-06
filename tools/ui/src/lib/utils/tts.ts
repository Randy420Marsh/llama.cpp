/**
 * Speech server client: text-to-speech, your own voices, model downloads and
 * speech-to-text.
 *
 * The server is tools/speech-server in this fork (any server with the same API
 * works). Its URL is a setting; empty means this page's own host on port 8179,
 * so the same default works on this PC and from a phone on the LAN.
 *
 *   POST /v1/audio/speech            {input, engine, voice, model, language, speed} -> WAV
 *   GET  /api/status                 engines, speech-to-text, model folders
 *   GET  /api/tts/engines            engines with their voices and models
 *   GET  /api/voices/library         your voices (reference clips for cloning)
 *   POST /api/voices/library         multipart {name, file, transcript, language}
 *   GET  /api/models                 models on disk and downloadable, per engine
 *   POST /api/models/download        {engine, id}
 *   POST /api/stt/transcribe         multipart {file, language} -> {text}
 */

export const SPEECH_DEFAULT_PORT = 8179;

// The first version of this UI defaulted to the full llama.cpp-server-tts app on
// 5001; a stored copy of that default now means "use the default" instead.
const LEGACY_DEFAULT_URLS = ['http://127.0.0.1:5001', 'http://localhost:5001'];

export function resolveSpeechUrl(configured: unknown): string {
	const raw = String(configured ?? '')
		.trim()
		.replace(/\/+$/, '');

	if (raw && !LEGACY_DEFAULT_URLS.includes(raw)) {
		return raw;
	}

	if (typeof window === 'undefined') {
		return `http://127.0.0.1:${SPEECH_DEFAULT_PORT}`;
	}

	const { hostname, protocol } = window.location;
	const host = hostname.includes(':') ? `[${hostname}]` : hostname;

	return `${protocol}//${host}:${SPEECH_DEFAULT_PORT}`;
}

export interface SpeechVoice {
	id: string;
	label: string;
	kind: 'preset' | 'custom';
	installed: boolean;
}

export interface SpeechModel {
	id: string;
	label: string;
	size_mb?: number;
	quant?: string;
	kind?: string;
}

export interface SpeechEngine {
	name: string;
	label: string;
	gpu: boolean;
	vram_mb: number;
	note: string;
	installed: boolean;
	detail: string;
	loaded: boolean;
	cloning: boolean;
	/** clones, but often misses the voice */
	cloning_experimental?: boolean;
	speed_native: boolean;
	languages: string[];
	voices: SpeechVoice[];
	default_voice: string;
	models: SpeechModel[];
	default_model: string;
}

// preferred order when no engine is chosen: llama.cpp (when it has a model), Kokoro, Piper
const AUTO_ENGINES = ['llamacpp', 'kokoro', 'piper'];

/** The engine a request will use: `want` when installed, else the best installed one. */
export function pickEngine(engines: SpeechEngine[], want: string): SpeechEngine | undefined {
	const usable = engines.filter((e) => e.installed);

	if (want) {
		const chosen = usable.find((e) => e.name === want);

		if (chosen) return chosen;
	}

	for (const name of AUTO_ENGINES) {
		const e = usable.find((x) => x.name === name && (x.name !== 'llamacpp' || x.models.length > 0));

		if (e) return e;
	}

	return usable[0];
}

export interface SpeechStatus {
	service: string;
	api: number;
	port: number;
	default_engine: string;
	offline: boolean;
	engines: { name: string; label: string; installed: boolean; loaded: boolean; gpu: boolean }[];
	stt: { installed: boolean; detail?: string; model?: string; server?: string; device?: string };
	paths: Record<string, string>;
	voices: number;
	reading_passage: string;
}

export interface LibraryVoice {
	id: string;
	name: string;
	language: string;
	transcript: string;
	duration_s: number;
	created: string;
	file: string;
	bytes: number;
}

export interface VoiceLibrary {
	voices: LibraryVoice[];
	dir: string;
	reading_passage: string;
	min_seconds: number;
	max_seconds: number;
}

export interface CatalogItem {
	id: string;
	label: string;
	size_mb: number;
	installed: boolean;
	recommended: boolean;
	note: string;
	source: string;
	dest: string;
}

export interface CatalogEngine {
	engine: string;
	label: string;
	installed: boolean;
	detail: string;
	items: CatalogItem[];
}

export interface DownloadStatus {
	running: boolean;
	engine: string;
	id: string;
	file: string;
	done: number;
	total: number;
	error: string;
	finished: boolean;
	started: number;
}

export interface ModelCatalog {
	paths: Record<string, string>;
	engines: CatalogEngine[];
	download: DownloadStatus;
	offline: boolean;
}

async function errorText(response: Response): Promise<string> {
	const raw = await response.text().catch(() => '');

	try {
		const doc = JSON.parse(raw);

		return doc?.error?.message || doc?.error || doc?.message || raw;
	} catch {
		return raw;
	}
}

async function request<T>(base: string, path: string, init?: RequestInit): Promise<T> {
	let response: Response;

	try {
		response = await fetch(`${base}${path}`, init);
	} catch {
		throw new Error(`speech server not reachable at ${base}`);
	}

	if (!response.ok) {
		const msg = await errorText(response);

		throw new Error(msg ? msg.slice(0, 300) : `HTTP ${response.status}`);
	}

	return (await response.json()) as T;
}

const json = (body: unknown): RequestInit => ({
	body: JSON.stringify(body),
	headers: { 'Content-Type': 'application/json' },
	method: 'POST'
});

export function getSpeechStatus(base: string): Promise<SpeechStatus> {
	return request<SpeechStatus>(base, '/api/status');
}

export async function getSpeechEngines(base: string): Promise<SpeechEngine[]> {
	return (await request<{ engines: SpeechEngine[] }>(base, '/api/tts/engines')).engines;
}

export function getModelCatalog(base: string): Promise<ModelCatalog> {
	return request<ModelCatalog>(base, '/api/models');
}

export function startModelDownload(base: string, engine: string, id: string) {
	return request<{ ok: boolean; error?: string }>(
		base,
		'/api/models/download',
		json({ engine, id })
	);
}

export function getDownloadStatus(base: string): Promise<DownloadStatus> {
	return request<DownloadStatus>(base, '/api/models/download/status');
}

export function cancelModelDownload(base: string): Promise<DownloadStatus> {
	return request<DownloadStatus>(base, '/api/models/download/cancel', { method: 'POST' });
}

export function unloadSpeechModels(base: string, engine = '') {
	return request<{ ok: boolean; unloaded: string[] }>(base, '/api/tts/unload', json({ engine }));
}

export function getVoiceLibrary(base: string): Promise<VoiceLibrary> {
	return request<VoiceLibrary>(base, '/api/voices/library');
}

export async function createVoice(
	base: string,
	voice: { name: string; file: Blob; filename: string; transcript?: string; language?: string }
): Promise<LibraryVoice> {
	const form = new FormData();

	form.append('name', voice.name);
	form.append('file', voice.file, voice.filename);
	form.append('transcript', voice.transcript ?? '');
	form.append('language', voice.language ?? '');

	return (
		await request<{ ok: boolean; voice: LibraryVoice }>(base, '/api/voices/library', {
			body: form,
			method: 'POST'
		})
	).voice;
}

export async function updateVoice(
	base: string,
	id: string,
	patch: { transcript?: string; language?: string; name?: string }
): Promise<LibraryVoice> {
	return (
		await request<{ ok: boolean; voice: LibraryVoice }>(
			base,
			`/api/voices/library/${encodeURIComponent(id)}`,
			{
				body: JSON.stringify(patch),
				headers: { 'Content-Type': 'application/json' },
				method: 'PATCH'
			}
		)
	).voice;
}

export function deleteVoice(base: string, id: string) {
	return request<{ ok: boolean }>(base, `/api/voices/library/${encodeURIComponent(id)}`, {
		method: 'DELETE'
	});
}

export function voiceAudioUrl(base: string, id: string): string {
	return `${base}/api/voices/library/${encodeURIComponent(id)}/audio`;
}

/**
 * Transcribe a recorded clip. Returns the trimmed transcript, or null when
 * there is nothing to transcribe or nothing was heard.
 */
export async function transcribeAudio(
	wavBlob: Blob,
	serverUrl: string,
	language = ''
): Promise<string | null> {
	const base = resolveSpeechUrl(serverUrl);

	if (wavBlob.size === 0) {
		return null;
	}

	const form = new FormData();

	form.append('file', wavBlob, 'clip.wav');
	form.append('language', language);

	const data = await request<{ text?: string }>(base, '/api/stt/transcribe', {
		body: form,
		method: 'POST'
	});
	const text = (data.text || '').trim();

	return text.length > 0 ? text : null;
}

export interface SynthesisRequest {
	engine?: string;
	voice?: string;
	model?: string;
	language?: string;
	speed?: number;
}

export interface SynthesisResult {
	blob: Blob;
	/** the server applied the speed itself; otherwise the player has to */
	speedNative: boolean;
	/** set when the requested engine failed and another one spoke instead */
	fallback: string;
	error: string;
}

export async function synthesize(
	base: string,
	text: string,
	req: SynthesisRequest,
	signal?: AbortSignal
): Promise<SynthesisResult> {
	let response: Response;

	try {
		response = await fetch(`${base}/v1/audio/speech`, {
			...json({
				engine: req.engine || undefined,
				input: text,
				language: req.language || undefined,
				model: req.model || undefined,
				speed: typeof req.speed === 'number' ? req.speed : 1,
				voice: req.voice || undefined
			}),
			signal
		});
	} catch (error) {
		if (signal?.aborted) {
			throw error;
		}

		throw new Error(`speech server not reachable at ${base}`);
	}

	if (!response.ok) {
		throw new Error((await errorText(response)).slice(0, 300) || `HTTP ${response.status}`);
	}

	return {
		blob: await response.blob(),
		error: response.headers.get('X-TTS-Error') || '',
		fallback: response.headers.get('X-TTS-Fallback') || '',
		speedNative: response.headers.get('X-TTS-Speed') === 'native'
	};
}

/** Reduce markdown to plain speech text: strip formatting, keep the words. */
export function toSpeechText(markdown: string): string {
	let out = markdown || '';

	// reasoning that leaked into the content is not part of the answer
	out = out.replace(/<think>[\s\S]*?<\/think>/g, '');
	// fenced code: keep the content, drop the fences and language tag
	out = out.replace(/```[\w+-]*\n?([\s\S]*?)```/g, '$1');
	out = out.replace(/`([^`]*)`/g, '$1');
	out = out.replace(/\*\*([^*]*)\*\*/g, '$1');
	out = out.replace(/__([^_]*)__/g, '$1');
	out = out.replace(/\*([^*\n]*)\*/g, '$1');
	out = out.replace(/~~([^~]*)~~/g, '$1');
	// images vanish, links keep their label, bare URLs are not read out
	out = out.replace(/!\[[^\]]*\]\([^)]*\)/g, '');
	out = out.replace(/\[([^\]]*)\]\((?:[^)]*)\)/g, '$1');
	out = out.replace(/https?:\/\/\S+/g, '');
	// headings, list bullets, blockquote markers, table rules and pipes
	// ([ \t], not \s: \s also eats the blank line before a list, and its pause)
	out = out.replace(/^[ \t]{0,3}#{1,6}[ \t]+/gm, '');
	out = out.replace(/^[ \t]*(?:[-*+]|\d+[.)])[ \t]+/gm, '');
	out = out.replace(/^[ \t]*>[ \t]?/gm, '');
	out = out.replace(/^[ \t]*\|?[ \t:|-]+\|?[ \t]*$/gm, '');
	out = out.replace(/\|/g, ' ');
	// emoji have no spoken form; engines read them as noise or their Unicode name
	out = out.replace(/\p{Extended_Pictographic}|[\u{1F1E6}-\u{1F1FF}]|\u{FE0F}|\u{200D}/gu, '');
	out = out.replace(/[ \t]+/g, ' ');
	out = out.replace(/ \n/g, '\n');
	out = out.replace(/\n{3,}/g, '\n\n');

	return out.trim();
}

/**
 * Split text into pieces to speak one after another. The first piece is kept
 * short so audio starts quickly; later pieces group sentences up to `max`
 * characters while the previous one plays.
 */
export function splitForSpeech(text: string, max = 280): string[] {
	const sentences = text
		.split(/(?<=[.!?。！？…])\s+|\n+/)
		.map((s) => s.trim())
		.filter((s) => /[\p{L}\p{N}]/u.test(s));
	const pieces: string[] = [];

	let current = '';

	for (const sentence of sentences) {
		const limit = pieces.length === 0 ? Math.min(max, 120) : max;

		if (current && (current + ' ' + sentence).length > limit) {
			pieces.push(current);
			current = sentence;
		} else {
			current = current ? `${current} ${sentence}` : sentence;
		}
	}

	if (current) {
		pieces.push(current);
	}

	// a single sentence longer than max is still one piece: the server splits it
	return pieces;
}
