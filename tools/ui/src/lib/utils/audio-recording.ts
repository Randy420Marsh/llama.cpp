import { MimeTypeAudio } from '$lib/enums';

/**
 * AudioRecorder - Browser-based audio recording with MediaRecorder API
 *
 * This class provides a complete audio recording solution using the browser's MediaRecorder API.
 * It handles microphone access, recording state management, and audio format optimization.
 *
 * **Features:**
 * - Automatic microphone permission handling
 * - Audio enhancement (echo cancellation, noise suppression, auto gain)
 * - Multiple format support with fallback (WAV, WebM, MP4, AAC)
 * - Real-time recording state tracking
 * - Proper cleanup and resource management
 */
/** Live microphone level sample (0..1 scale, peak-hold + clip flag). */
export interface MicLevelSample {
	/** Current level, 0..1 (mapped from -60..0 dBFS). */
	level: number;
	/** Peak-hold level, decays over ~1s, 0..1. */
	peak: number;
	/** True when the peak reached full scale. */
	clip: boolean;
}

/** Options for starting a recording with gain, noise reduction, metering and VAD auto-stop. */
export interface AudioRecorderOptions {
	/** Input gain, 1 = unchanged (up to 3x amplification). */
	gain?: number;
	/** Apply noise suppression + echo cancellation constraints (default true). */
	noiseCancelling?: boolean;
	/** Called ~20x/sec with the input level meter data while recording. */
	onLevel?: (sample: MicLevelSample) => void;
	/** Called once when the silence timeout elapses (voice activity detection auto-stop). */
	onAutoStop?: () => void;
	/** Silence (ms) that triggers onAutoStop. Default 5000. */
	autoStopSilenceMs?: number;
}

export class AudioRecorder {
	private analyser: AnalyserNode | null = null;
	private audioChunks: Blob[] = [];
	private audioContext: AudioContext | null = null;
	private destStream: MediaStream | null = null;
	private gain = 1;
	private gainNode: GainNode | null = null;
	private lastSpeechTime = 0;
	private mediaRecorder: MediaRecorder | null = null;
	private meterData: Float32Array<ArrayBuffer> | null = null;
	private meterTimer: number | null = null;
	private options: AudioRecorderOptions = {};
	private peakLevel = 0;
	private recordingState: boolean = false;
	private stream: MediaStream | null = null;

	cancelRecording(): void {
		const recorder = this.mediaRecorder;
		const stream = this.stream;

		this.mediaRecorder = null;
		this.audioChunks = [];
		this.stream = null;
		this.recordingState = false;
		this.cleanupAudioGraph();

		if (recorder && recorder.state !== 'inactive') {
			// Drop the original handlers so the pending stop event does not touch the instance
			recorder.onstop = null;
			recorder.onerror = null;
			recorder.stop();
		}

		if (stream) {
			for (const track of stream.getTracks()) {
				track.stop();
			}
		}
	}

	isRecording(): boolean {
		return this.recordingState;
	}

	/** Live input gain, applied to an in-flight recording without restarting it. */
	setGain(gain: number): void {
		this.gain = gain;

		if (this.gainNode && this.audioContext) {
			this.gainNode.gain.setTargetAtTime(gain, this.audioContext.currentTime, 0.05);
		}
	}

	/** Toggle noise suppression + echo cancellation on the live mic track. */
	async setNoiseCancelling(enabled: boolean): Promise<void> {
		const track = this.stream?.getAudioTracks()[0];

		if (!track) {
			return;
		}

		try {
			await track.applyConstraints({
				echoCancellation: true,
				noiseSuppression: enabled
			});
		} catch (error) {
			console.warn('Failed to apply noise constraints', error);
		}
	}

	async startRecording(options: AudioRecorderOptions = {}): Promise<void> {
		try {
			this.options = options;
			const nr = options.noiseCancelling !== false;
			const gain = options.gain ?? this.gain;

			this.gain = gain;

			this.stream = await navigator.mediaDevices.getUserMedia({
				audio: {
					// Leave AGC to the OS when no explicit gain is requested; an explicit gain
					// goes through our own GainNode instead, so AGC would fight it.
					autoGainControl: gain === 1,
					echoCancellation: true,
					noiseSuppression: nr
				}
			});

			this.initializeAudioGraph(this.stream, gain);
			this.startMetering();

			// Record from the processed graph (gain + metering) when available, else the raw stream.
			this.initializeRecorder(this.destStream ?? this.stream);

			this.audioChunks = [];
			// Start recording with a small timeslice to ensure we get data
			this.mediaRecorder!.start(100);
			this.recordingState = true;
		} catch (error) {
			console.error('Failed to start recording:', error);
			this.cleanupAudioGraph();

			throw new Error('Failed to access microphone. Please check permissions.');
		}
	}

	async stopRecording(): Promise<Blob> {
		return new Promise((resolve, reject) => {
			const recorder = this.mediaRecorder;
			const chunks = this.audioChunks;
			const stream = this.stream;

			if (!recorder || recorder.state === 'inactive') {
				reject(new Error('No active recording to stop'));

				return;
			}

			// Detach instance state right away so a new startRecording can take over without race
			this.mediaRecorder = null;
			this.audioChunks = [];
			this.stream = null;
			this.recordingState = false;

			recorder.onstop = () => {
				const audioBlob = new Blob(chunks, {
					type: recorder.mimeType || MimeTypeAudio.WAV
				});

				if (stream) {
					for (const track of stream.getTracks()) {
						track.stop();
					}
				}

				this.cleanupAudioGraph();
				resolve(audioBlob);
			};

			recorder.onerror = (event) => {
				console.error('Recording error:', event);

				if (stream) {
					for (const track of stream.getTracks()) {
						track.stop();
					}
				}

				this.cleanupAudioGraph();
				reject(new Error('Recording failed'));
			};

			recorder.stop();
		});
	}

	private cleanupAudioGraph(): void {
		if (this.meterTimer !== null) {
			window.clearInterval(this.meterTimer);
			this.meterTimer = null;
		}

		this.meterData = null;
		this.peakLevel = 0;

		if (this.audioContext) {
			this.audioContext.close().catch(() => {});
			this.audioContext = null;
		}

		this.gainNode = null;
		this.analyser = null;
		this.destStream = null;
	}

	/**
	 * Build the Web Audio graph: mic -> GainNode -> AnalyserNode -> MediaStreamDestination.
	 * The destination stream feeds the MediaRecorder (so the gain applies to the recording);
	 * the analyser drives the level meter + VAD. Falls back to the raw stream on failure.
	 */
	private initializeAudioGraph(stream: MediaStream, gain: number): void {
		try {
			const Ctor =
				window.AudioContext ??
				(window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;

			if (!Ctor) {
				return;
			}

			this.audioContext = new Ctor();
			const source = this.audioContext.createMediaStreamSource(stream);

			this.gainNode = this.audioContext.createGain();
			this.gainNode.gain.value = gain;
			this.analyser = this.audioContext.createAnalyser();
			this.analyser.fftSize = 1024;

			// createMediaStreamDestination is the spec name; the alias is kept for older browsers.
			const dest =
				typeof this.audioContext.createMediaStreamDestination === 'function'
					? this.audioContext.createMediaStreamDestination()
					: (
							this.audioContext as unknown as {
								createMediaStreamAudioDestinationNode(): MediaStreamAudioDestinationNode;
							}
						).createMediaStreamAudioDestinationNode();

			source.connect(this.gainNode);
			this.gainNode.connect(this.analyser);
			this.gainNode.connect(dest);
			this.destStream = dest.stream;
		} catch (error) {
			console.warn('Web Audio graph unavailable, recording from raw stream', error);
			this.cleanupAudioGraph();
		}
	}

	private initializeRecorder(stream: MediaStream): void {
		const options: MediaRecorderOptions = {};

		if (MediaRecorder.isTypeSupported(MimeTypeAudio.WAV)) {
			options.mimeType = MimeTypeAudio.WAV;
		} else if (MediaRecorder.isTypeSupported(MimeTypeAudio.WEBM_OPUS)) {
			options.mimeType = MimeTypeAudio.WEBM_OPUS;
		} else if (MediaRecorder.isTypeSupported(MimeTypeAudio.WEBM)) {
			options.mimeType = MimeTypeAudio.WEBM;
		} else if (MediaRecorder.isTypeSupported(MimeTypeAudio.MP4)) {
			options.mimeType = MimeTypeAudio.MP4;
		} else {
			console.warn('No preferred audio format supported, using default');
		}

		this.mediaRecorder = new MediaRecorder(stream, options);

		this.mediaRecorder.ondataavailable = (event) => {
			if (event.data.size > 0) {
				this.audioChunks.push(event.data);
			}
		};

		this.mediaRecorder.onstop = () => {
			this.recordingState = false;
		};

		this.mediaRecorder.onerror = (event) => {
			console.error('MediaRecorder error:', event);
			this.recordingState = false;
		};
	}

	private startMetering(): void {
		if (!this.analyser || this.meterTimer !== null) {
			return;
		}

		this.meterData = new Float32Array(this.analyser.fftSize);
		this.peakLevel = 0;
		this.lastSpeechTime = performance.now();

		this.meterTimer = window.setInterval(() => this.updateMeter(), 50);
	}

	private updateMeter(): void {
		const analyser = this.analyser;
		const data = this.meterData;

		if (!analyser || !data) {
			return;
		}

		analyser.getFloatTimeDomainData(data);

		let sum = 0;

		for (let i = 0; i < data.length; i += 1) {
			sum += data[i] * data[i];
		}

		const rms = Math.sqrt(sum / data.length);
		const db = rms > 0 ? 20 * Math.log10(rms) : -60;
		const level = Math.min(1, Math.max(0, (db + 60) / 60));

		if (level > this.peakLevel) {
			this.peakLevel = level;
		} else {
			// Peak-hold decay: ~0.03/tick at 20 Hz clears the peak in roughly a second.
			this.peakLevel = Math.max(0, this.peakLevel - 0.03);
		}

		this.options.onLevel?.({ clip: this.peakLevel >= 0.99, level, peak: this.peakLevel });

		// Voice activity detection: stop automatically after the configured silence.
		const now = performance.now();

		if (db > -45) {
			this.lastSpeechTime = now;
		} else if (
			this.options.onAutoStop &&
			now - this.lastSpeechTime >= (this.options.autoStopSilenceMs ?? 5000)
		) {
			const onAutoStop = this.options.onAutoStop;

			this.options.onAutoStop = undefined;
			onAutoStop();
		}
	}
}

export async function convertToWav(audioBlob: Blob): Promise<Blob> {
	try {
		if (audioBlob.type.includes('wav')) {
			return audioBlob;
		}

		const arrayBuffer = await audioBlob.arrayBuffer();
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();

		try {
			const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

			return audioBufferToWav(audioBuffer);
		} finally {
			audioContext.close();
		}
	} catch (error) {
		console.error('Failed to convert audio to WAV:', error);

		return audioBlob;
	}
}

function audioBufferToWav(buffer: AudioBuffer): Blob {
	const length = buffer.length;
	const numberOfChannels = buffer.numberOfChannels;
	const sampleRate = buffer.sampleRate;
	const bytesPerSample = 2; // 16-bit
	const blockAlign = numberOfChannels * bytesPerSample;
	const byteRate = sampleRate * blockAlign;
	const dataSize = length * blockAlign;
	const bufferSize = 44 + dataSize;
	const arrayBuffer = new ArrayBuffer(bufferSize);
	const view = new DataView(arrayBuffer);
	const writeString = (offset: number, string: string) => {
		for (let i = 0; i < string.length; i++) {
			view.setUint8(offset + i, string.charCodeAt(i));
		}
	};

	writeString(0, 'RIFF'); // ChunkID
	view.setUint32(4, bufferSize - 8, true); // ChunkSize
	writeString(8, 'WAVE'); // Format
	writeString(12, 'fmt '); // Subchunk1ID
	view.setUint32(16, 16, true); // Subchunk1Size
	view.setUint16(20, 1, true); // AudioFormat (PCM)
	view.setUint16(22, numberOfChannels, true); // NumChannels
	view.setUint32(24, sampleRate, true); // SampleRate
	view.setUint32(28, byteRate, true); // ByteRate
	view.setUint16(32, blockAlign, true); // BlockAlign
	view.setUint16(34, 16, true); // BitsPerSample
	writeString(36, 'data'); // Subchunk2ID
	view.setUint32(40, dataSize, true); // Subchunk2Size

	// Cache channel arrays, write PCM via Int16Array (native little-endian, matches WAV)
	const channels: Float32Array[] = new Array(numberOfChannels);

	for (let c = 0; c < numberOfChannels; c++) {
		channels[c] = buffer.getChannelData(c);
	}

	const pcm = new Int16Array(arrayBuffer, 44, length * numberOfChannels);

	let p = 0;

	for (let i = 0; i < length; i++) {
		for (let c = 0; c < numberOfChannels; c++) {
			let s = channels[c][i];

			if (s > 1) s = 1;
			else if (s < -1) s = -1;

			pcm[p++] = s * 0x7fff;
		}
	}

	return new Blob([arrayBuffer], { type: MimeTypeAudio.WAV });
}

/**
 * Create a File object from audio blob with timestamp-based naming
 * @param audioBlob - The audio blob to wrap
 * @param filename - Optional custom filename
 * @returns File object with appropriate name and metadata
 */
export function createAudioFile(audioBlob: Blob, filename?: string): File {
	const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
	const extension = audioBlob.type.includes('wav') ? 'wav' : 'mp3';
	const defaultFilename = `recording-${timestamp}.${extension}`;

	return new File([audioBlob], filename || defaultFilename, {
		lastModified: Date.now(),
		type: audioBlob.type
	});
}

/**
 * Check if audio recording is supported in the current browser
 * @returns True if MediaRecorder and getUserMedia are available
 */
export function isAudioRecordingSupported(): boolean {
	return !!(
		typeof navigator !== 'undefined' &&
		navigator.mediaDevices &&
		typeof navigator.mediaDevices.getUserMedia === 'function' &&
		typeof window !== 'undefined' &&
		window.MediaRecorder
	);
}
