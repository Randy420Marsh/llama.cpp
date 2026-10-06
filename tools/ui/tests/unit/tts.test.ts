import {
	pickEngine,
	resolveSpeechUrl,
	type SpeechEngine,
	splitForSpeech,
	toSpeechText
} from '$lib/utils/tts';
import { describe, expect, it } from 'vitest';

const engine = (name: string, installed = true, models = 0): SpeechEngine => ({
	cloning: false,
	default_model: '',
	default_voice: '',
	detail: '',
	gpu: false,
	installed,
	label: name,
	languages: [],
	loaded: false,
	models: Array.from({ length: models }, (_, i) => ({ id: `m${i}`, label: `m${i}` })),
	name,
	note: '',
	speed_native: false,
	voices: [],
	vram_mb: 0
});

describe('resolveSpeechUrl', () => {
	it('keeps an explicit URL without the trailing slash', () => {
		expect(resolveSpeechUrl('http://10.0.0.2:9000/')).toBe('http://10.0.0.2:9000');
	});

	it('treats empty and the old 5001 default as "this host, port 8179"', () => {
		const expected = resolveSpeechUrl('');

		expect(expected.endsWith(':8179')).toBe(true);
		expect(resolveSpeechUrl('http://127.0.0.1:5001')).toBe(expected);
		expect(resolveSpeechUrl(undefined)).toBe(expected);
	});
});

describe('pickEngine', () => {
	const all = [engine('piper'), engine('kokoro', false), engine('llamacpp', true, 1)];

	it('uses the chosen engine when it is installed', () => {
		expect(pickEngine(all, 'piper')?.name).toBe('piper');
	});

	it('falls back to the best installed engine', () => {
		expect(pickEngine(all, '')?.name).toBe('llamacpp');
		expect(pickEngine(all, 'kokoro')?.name).toBe('llamacpp');
	});

	it('skips the llama.cpp engine while it has no model', () => {
		expect(pickEngine([engine('llamacpp', true, 0), engine('piper')], '')?.name).toBe('piper');
	});

	it('returns undefined when nothing is installed', () => {
		expect(pickEngine([engine('piper', false)], '')).toBeUndefined();
	});
});

describe('toSpeechText', () => {
	it('drops markdown syntax, links targets, URLs and emoji but keeps the words', () => {
		const md =
			'# Title\n\n**Bold** and *italic* with `code`, a [link](http://x.y) and http://example.com 🌊\n\n- item one\n- item two';

		expect(toSpeechText(md)).toBe(
			'Title\n\nBold and italic with code, a link and\n\nitem one\nitem two'
		);
	});

	it('removes reasoning that leaked into the content', () => {
		expect(toSpeechText('<think>hidden</think>Answer.')).toBe('Answer.');
	});

	it('keeps the inside of fenced code', () => {
		expect(toSpeechText('Run:\n```bash\necho hi\n```')).toBe('Run:\necho hi');
	});
});

describe('splitForSpeech', () => {
	it('keeps the first piece short so audio starts quickly', () => {
		const text = 'First sentence here. ' + 'Another fairly long sentence follows it. '.repeat(10);
		const pieces = splitForSpeech(text);

		expect(pieces[0].startsWith('First sentence here.')).toBe(true);
		expect(pieces[0].length).toBeLessThanOrEqual(120);
		expect(pieces.length).toBeGreaterThan(1);
		expect(pieces.slice(1).every((p) => p.length <= 280)).toBe(true);
	});

	it('drops pieces with nothing pronounceable and merges short sentences', () => {
		expect(splitForSpeech('Hello.\n---\n. . .\nBye.')).toEqual(['Hello. Bye.']);
	});
});
