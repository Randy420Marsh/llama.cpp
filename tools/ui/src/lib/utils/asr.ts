import { apiFetch } from './api-fetch';

/**
 * Transcribe a recorded clip with llama-server's own --asr-url whisper
 * (POST /asr/transcribe). Returns the trimmed transcript, or null when nothing
 * was heard. `model` routes the request on a router.
 */
export async function transcribeWithServer(
	wavBlob: Blob,
	model: string | null | undefined,
	language = ''
): Promise<string | null> {
	if (wavBlob.size === 0) {
		return null;
	}

	const bytes = new Uint8Array(await wavBlob.arrayBuffer());

	let binary = '';

	for (let i = 0; i < bytes.length; i += 0x8000) {
		binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
	}

	const data = await apiFetch<{ text?: string }>('/asr/transcribe', {
		body: JSON.stringify({ audio: btoa(binary), language, model: model || undefined }),
		method: 'POST'
	});
	const text = (data.text || '').trim();

	return text.length > 0 ? text : null;
}
