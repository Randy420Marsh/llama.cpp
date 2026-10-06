<script lang="ts">
	/**
	 * Settings → Voice: everything the speech server offers, as lists.
	 *
	 *   read aloud   engine, voice, model and language (GET /api/tts/engines)
	 *   your voices  record or upload a clip; the cloning engines speak with it
	 *   models       on disk and downloadable, per engine, with the folder each
	 *                one belongs in (GET /api/models)
	 *
	 * Choices go into localConfig and are kept by the dialog's Save button;
	 * voices and downloads act on the server straight away.
	 */
	import {
		Check,
		CircleAlert,
		Download,
		FolderOpen,
		LoaderCircle,
		Mic,
		Pencil,
		Play,
		Power,
		RotateCw,
		Square,
		Trash2,
		Upload,
		X
	} from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import * as Select from '$lib/components/ui/select';
	import { Textarea } from '$lib/components/ui/textarea';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { speechStore } from '$lib/stores';
	import { AudioRecorder, convertToWav, isAudioRecordingSupported } from '$lib/utils/browser-only';
	import {
		cancelModelDownload,
		type CatalogEngine,
		createVoice,
		deleteVoice,
		type DownloadStatus,
		getDownloadStatus,
		getModelCatalog,
		getSpeechEngines,
		getSpeechStatus,
		getVoiceLibrary,
		type LibraryVoice,
		pickEngine,
		resolveSpeechUrl,
		type SpeechEngine,
		type SpeechStatus,
		startModelDownload,
		unloadSpeechModels,
		updateVoice,
		voiceAudioUrl
	} from '$lib/utils/tts';
	import { onDestroy } from 'svelte';
	import { toast } from 'svelte-sonner';

	interface Props {
		localConfig: SettingsConfigType;
		onConfigChange: (key: string, value: string | number | boolean) => void;
	}

	let { localConfig, onConfigChange }: Props = $props();

	// bits-ui treats an empty value as "nothing selected", so "use the default" needs a name
	const AUTO = '__auto__';
	const TEST_ID = 'settings-voice-test';
	const LANG_NAMES: Record<string, string> = {
		de: 'German',
		en: 'English',
		es: 'Spanish',
		fi: 'Finnish',
		fr: 'French',
		it: 'Italian',
		ja: 'Japanese',
		ko: 'Korean',
		pt: 'Portuguese',
		ru: 'Russian',
		sv: 'Swedish',
		zh: 'Chinese'
	};
	// where each engine's files go, keyed by the server's folder names
	const ENGINE_DIR: Record<string, string> = {
		chatterbox: 'hf',
		kokoro: 'hf',
		llamacpp: 'native_tts',
		orpheus: 'orpheus',
		piper: 'piper'
	};

	const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));
	const langName = (code: string) => LANG_NAMES[code] ?? code;

	// ---------------------------------------------------------------- server state
	let base = $derived(resolveSpeechUrl(localConfig[SETTINGS_KEYS.TTS_SERVER_URL]));

	let status = $state<SpeechStatus | null>(null);
	let engines = $state<SpeechEngine[]>([]);
	let library = $state<LibraryVoice[]>([]);
	let catalog = $state<CatalogEngine[]>([]);
	let paths = $state<Record<string, string>>({});
	let passage = $state('');
	let minSeconds = $state(3);
	let maxSeconds = $state(30);
	let loadError = $state('');
	let loading = $state(false);
	let loadedFor = '';

	async function load(url = base) {
		loading = true;

		try {
			const [st, en, lib, cat] = await Promise.all([
				getSpeechStatus(url),
				getSpeechEngines(url),
				getVoiceLibrary(url),
				getModelCatalog(url)
			]);

			if (url !== base) return;

			status = st;
			engines = en;
			library = lib.voices;
			passage = lib.reading_passage || st.reading_passage;
			minSeconds = lib.min_seconds;
			maxSeconds = lib.max_seconds;
			catalog = cat.engines;
			paths = cat.paths;
			loadError = '';

			if (cat.download.running) {
				download = cat.download;
				pollDownload();
			}
		} catch (error) {
			if (url !== base) return;

			status = null;
			engines = [];
			library = [];
			catalog = [];
			loadError = msg(error);
		} finally {
			loading = false;
		}
	}

	// (re)load when the URL changes, debounced while it is being typed
	let debounce: ReturnType<typeof setTimeout> | undefined;

	$effect(() => {
		const url = base;

		if (url === loadedFor) return;

		loadedFor = url;
		clearTimeout(debounce);
		debounce = setTimeout(() => void load(url), status ? 500 : 0);
	});

	// ---------------------------------------------------------------- read aloud
	let engineSetting = $derived(String(localConfig[SETTINGS_KEYS.TTS_ENGINE] ?? ''));
	let engine = $derived(pickEngine(engines, engineSetting));
	let voiceSetting = $derived(String(localConfig[SETTINGS_KEYS.TTS_VOICE] ?? ''));
	let voiceValid = $derived(!!engine?.voices.some((v) => v.id === voiceSetting));
	let customVoices = $derived(engine?.voices.filter((v) => v.kind === 'custom') ?? []);
	let presetVoices = $derived(engine?.voices.filter((v) => v.kind !== 'custom') ?? []);
	let modelSetting = $derived(String(localConfig[SETTINGS_KEYS.TTS_MODEL] ?? ''));
	let modelValid = $derived(!!engine?.models.some((m) => m.id === modelSetting));
	let languageSetting = $derived(
		String(localConfig[SETTINGS_KEYS.TTS_LANGUAGE] ?? 'auto') || 'auto'
	);
	let speed = $derived(Number(localConfig[SETTINGS_KEYS.TTS_SPEED] ?? 1) || 1);

	let engineLabel = $derived(
		engineSetting && engine?.name === engineSetting
			? engine.label
			: `Automatic${engine ? ` (${engine.label})` : ''}`
	);
	let voiceLabel = $derived.by(() => {
		if (!engine) return 'No engine';

		const v = engine.voices.find((x) => x.id === voiceSetting);

		if (v) return v.label;

		const d = engine.voices.find((x) => x.id === engine?.default_voice);

		return `Engine default${d ? ` (${d.label})` : ''}`;
	});
	let modelLabel = $derived.by(() => {
		if (!engine) return '';

		const m = engine.models.find((x) => x.id === modelSetting);

		if (m) return m.label;

		const d = engine.models.find((x) => x.id === engine?.default_model);

		return `Default${d ? ` (${d.label})` : ''}`;
	});

	function setEngine(value: string) {
		onConfigChange(SETTINGS_KEYS.TTS_ENGINE, value === AUTO ? '' : value);
		// voices and models belong to one engine
		onConfigChange(SETTINGS_KEYS.TTS_VOICE, '');
		onConfigChange(SETTINGS_KEYS.TTS_MODEL, '');
	}

	let testText = $state('Hello! This is how my replies will sound with this voice.');
	let testing = $derived(speechStore.speakingId === TEST_ID);

	function test() {
		if (testing) {
			speechStore.stop();

			return;
		}

		void speechStore.speak(
			TEST_ID,
			testText,
			{
				engine: engine?.name,
				language: languageSetting,
				model: modelValid ? modelSetting : '',
				speed,
				voice: voiceValid ? voiceSetting : ''
			},
			base
		);
	}

	// ---------------------------------------------------------------- your voices
	let recorder: AudioRecorder | null = null;
	let recState = $state<'idle' | 'recording' | 'ready'>('idle');
	let recSeconds = $state(0);
	let recLevel = $state(0);
	let recTimer: ReturnType<typeof setInterval> | undefined;
	let clip = $state<Blob | null>(null);
	let clipName = $state('');
	let clipUrl = $state('');
	let newName = $state('');
	let newLanguage = $state('');
	let newTranscript = $state('');
	let saving = $state(false);
	let fileInput = $state<HTMLInputElement | null>(null);
	let editing = $state<string | null>(null);
	let editText = $state('');
	let preview: HTMLAudioElement | null = null;
	let previewing = $state<string | null>(null);

	function setClip(blob: Blob, name: string) {
		if (clipUrl) URL.revokeObjectURL(clipUrl);

		clip = blob;
		clipName = name;
		clipUrl = URL.createObjectURL(blob);
		recState = 'ready';
	}

	function discardClip() {
		if (clipUrl) URL.revokeObjectURL(clipUrl);

		clip = null;
		clipName = '';
		clipUrl = '';
		recState = 'idle';
	}

	async function startRecording() {
		discardClip();

		try {
			recorder = new AudioRecorder();
			await recorder.startRecording({
				gain: 1,
				noiseCancelling: true,
				onLevel: (s) => (recLevel = s.level)
			});
			recState = 'recording';
			recSeconds = 0;
			recTimer = setInterval(() => {
				recSeconds += 1;

				if (recSeconds >= maxSeconds) void stopRecording();
			}, 1000);

			// the passage on screen is what will be said
			if (!newTranscript.trim()) newTranscript = passage;
		} catch (error) {
			recorder = null;
			toast.error(`Microphone: ${msg(error)}`);
		}
	}

	async function stopRecording() {
		clearInterval(recTimer);
		recTimer = undefined;

		const r = recorder;

		recorder = null;
		recLevel = 0;

		if (!r) return;

		try {
			const wav = await convertToWav(await r.stopRecording());

			setClip(wav, 'recording.wav');
		} catch (error) {
			recState = 'idle';
			toast.error(`Recording failed: ${msg(error)}`);
		}
	}

	function pickFile() {
		const f = fileInput?.files?.[0];

		if (!f) return;

		setClip(f, f.name);

		if (newTranscript === passage) newTranscript = '';

		if (!newName.trim()) newName = f.name.replace(/\.[^.]+$/, '');
	}

	async function saveVoice() {
		if (!clip || !newName.trim()) return;

		saving = true;

		try {
			const v = await createVoice(base, {
				file: clip,
				filename: clipName,
				language: newLanguage.trim(),
				name: newName.trim(),
				transcript: newTranscript.trim()
			});

			toast.success(`Voice "${v.name}" saved (${v.duration_s.toFixed(1)} s)`);
			discardClip();
			newName = '';
			newTranscript = '';
			await load();

			// speak with it right away: keep a cloning engine, else switch to one
			const target = engine?.cloning
				? engine
				: (engines.find((e) => e.name === 'llamacpp' && e.installed && e.models.length > 0) ??
					engines.find((e) => e.cloning && e.installed));

			if (target) {
				onConfigChange(SETTINGS_KEYS.TTS_ENGINE, target.name);
				onConfigChange(SETTINGS_KEYS.TTS_VOICE, v.id);
			}

			void speechStore.refresh();
		} catch (error) {
			toast.error(`Could not save the voice: ${msg(error)}`);
		} finally {
			saving = false;
		}
	}

	function playReference(id: string) {
		preview?.pause();

		if (previewing === id) {
			previewing = null;

			return;
		}

		preview = new Audio(voiceAudioUrl(base, id));
		previewing = id;
		preview.onended = () => (previewing = null);
		preview.onerror = () => (previewing = null);
		void preview.play().catch(() => (previewing = null));
	}

	async function removeVoice(v: LibraryVoice) {
		if (!confirm(`Delete the voice "${v.name}"? Its reference clip is removed from the server.`)) {
			return;
		}

		try {
			await deleteVoice(base, v.id);

			if (voiceSetting === v.id) onConfigChange(SETTINGS_KEYS.TTS_VOICE, '');

			await load();
			void speechStore.refresh();
		} catch (error) {
			toast.error(`Could not delete: ${msg(error)}`);
		}
	}

	async function saveTranscript(v: LibraryVoice) {
		try {
			await updateVoice(base, v.id, { transcript: editText });
			editing = null;
			await load();
		} catch (error) {
			toast.error(`Could not save: ${msg(error)}`);
		}
	}

	// ---------------------------------------------------------------- models
	let download = $state<DownloadStatus | null>(null);
	let pollTimer: ReturnType<typeof setInterval> | undefined;
	let openEngines = $state<Record<string, boolean>>({});
	let unloading = $state(false);

	let downloadPct = $derived(
		download && download.total > 0 ? Math.min(100, (download.done / download.total) * 100) : 0
	);

	function pollDownload() {
		clearInterval(pollTimer);
		pollTimer = setInterval(async () => {
			try {
				const s = await getDownloadStatus(base);

				download = s;

				if (!s.running) {
					clearInterval(pollTimer);
					pollTimer = undefined;

					if (s.error) {
						toast.error(`Download failed: ${s.error}`);
					} else if (s.finished) {
						toast.success(`Downloaded ${s.engine} / ${s.id}`);
					}

					await load();
					void speechStore.refresh();
				}
			} catch {
				// the server may be busy writing; keep polling
			}
		}, 1500);
	}

	async function startDownload(engineName: string, id: string) {
		try {
			await startModelDownload(base, engineName, id);
			download = {
				done: 0,
				engine: engineName,
				error: '',
				file: '',
				finished: false,
				id,
				running: true,
				started: Date.now() / 1000,
				total: 0
			};
			pollDownload();
		} catch (error) {
			toast.error(`Download not started: ${msg(error)}`);
		}
	}

	async function cancelDownload() {
		try {
			download = await cancelModelDownload(base);
		} catch (error) {
			toast.error(msg(error));
		}
	}

	async function unload() {
		unloading = true;

		try {
			const r = await unloadSpeechModels(base);

			toast.success(
				r.unloaded.length ? `Unloaded: ${r.unloaded.join(', ')}` : 'Nothing was loaded'
			);
			await load();
		} catch (error) {
			toast.error(msg(error));
		} finally {
			unloading = false;
		}
	}

	const mb = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)} GB` : `${Math.round(n)} MB`);

	onDestroy(() => {
		clearTimeout(debounce);
		clearInterval(pollTimer);
		clearInterval(recTimer);
		recorder?.cancelRecording();
		preview?.pause();

		if (clipUrl) URL.revokeObjectURL(clipUrl);
	});
</script>

<div class="space-y-8">
	<!-- connection -->
	<div
		class="flex items-start justify-between gap-3 rounded-lg border border-border/40 px-3 py-2.5 text-sm"
	>
		<div class="min-w-0 space-y-0.5">
			{#if status}
				<div class="flex items-center gap-2 font-medium">
					<span class="h-2 w-2 shrink-0 rounded-full bg-emerald-500"></span>
					Speech server connected
				</div>

				<p class="truncate text-xs text-muted-foreground">
					{base} · {engines.filter((e) => e.installed).length} of {engines.length} engines ready · speech-to-text:
					{status.stt.server
						? `ASR server (${status.stt.server})`
						: status.stt.installed
							? `local ${status.stt.model ?? 'Whisper'} (${status.stt.device ?? 'cpu'})`
							: 'not available'}
				</p>
			{:else if loading}
				<div class="flex items-center gap-2 text-muted-foreground">
					<LoaderCircle class="h-3.5 w-3.5 animate-spin" />
					Contacting {base}…
				</div>
			{:else}
				<div class="flex items-center gap-2 font-medium text-destructive">
					<CircleAlert class="h-4 w-4 shrink-0" />
					No speech server at {base}
				</div>

				<p class="text-xs text-muted-foreground">
					Start it with <code>tools/speech-server/run.bat --no-https --lan</code> (or
					<code>run.sh</code>), or set its URL above. {loadError}
				</p>
			{/if}
		</div>

		<Button class="shrink-0" onclick={() => load()} size="sm" variant="ghost">
			<RotateCw class="h-3.5 w-3.5 {loading ? 'animate-spin' : ''}" />
			Refresh
		</Button>
	</div>

	{#if status}
		<!-- read aloud -->
		<section class="space-y-4">
			<h3 class="text-sm font-semibold">Read aloud</h3>

			<div class="grid gap-4 sm:grid-cols-2">
				<div class="space-y-2">
					<Label class="text-sm font-medium">Engine</Label>

					<Select.Root
						onValueChange={(v) => setEngine(v)}
						type="single"
						value={engineSetting && engine?.name === engineSetting ? engineSetting : AUTO}
					>
						<Select.Trigger class="w-full">{engineLabel}</Select.Trigger>

						<Select.Content>
							<Select.Item label="Automatic" value={AUTO}>Automatic (best installed)</Select.Item>

							{#each engines as e (e.name)}
								<Select.Item disabled={!e.installed} label={e.label} value={e.name}>
									<div class="flex flex-col">
										<span>
											{e.label}{e.cloning
												? e.cloning_experimental
													? ' · clones voices (experimental)'
													: ' · clones voices'
												: ''}
										</span>

										<span class="text-xs text-muted-foreground">
											{e.installed ? e.note : e.detail}
										</span>
									</div>
								</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
				</div>

				<div class="space-y-2">
					<Label class="text-sm font-medium">Voice</Label>

					<Select.Root
						disabled={!engine}
						onValueChange={(v) => onConfigChange(SETTINGS_KEYS.TTS_VOICE, v === AUTO ? '' : v)}
						type="single"
						value={voiceValid ? voiceSetting : AUTO}
					>
						<Select.Trigger class="w-full">{voiceLabel}</Select.Trigger>

						<Select.Content>
							<Select.Item label="Engine default" value={AUTO}>Engine default</Select.Item>

							{#if customVoices.length}
								<Select.Group>
									<Select.GroupHeading>Your voices</Select.GroupHeading>

									{#each customVoices as v (v.id)}
										<Select.Item label={v.label} value={v.id}>{v.label}</Select.Item>
									{/each}
								</Select.Group>
							{/if}

							{#if presetVoices.length}
								<Select.Group>
									<Select.GroupHeading>Built-in</Select.GroupHeading>

									{#each presetVoices as v (v.id)}
										<Select.Item disabled={!v.installed} label={v.label} value={v.id}>
											{v.label}{v.installed ? '' : ' (not downloaded)'}
										</Select.Item>
									{/each}
								</Select.Group>
							{/if}
						</Select.Content>
					</Select.Root>
				</div>

				{#if engine && engine.models.length > 0}
					<div class="space-y-2">
						<Label class="text-sm font-medium">Model</Label>

						<Select.Root
							onValueChange={(v) => onConfigChange(SETTINGS_KEYS.TTS_MODEL, v === AUTO ? '' : v)}
							type="single"
							value={modelValid ? modelSetting : AUTO}
						>
							<Select.Trigger class="w-full">{modelLabel}</Select.Trigger>

							<Select.Content>
								<Select.Item label="Default" value={AUTO}>Default</Select.Item>

								{#each engine.models as m (m.id)}
									<Select.Item label={m.label} value={m.id}>
										{m.label}{m.size_mb ? ` · ${mb(m.size_mb)}` : ''}
									</Select.Item>
								{/each}
							</Select.Content>
						</Select.Root>
					</div>
				{/if}

				{#if engine && engine.languages.length > 0}
					<div class="space-y-2">
						<Label class="text-sm font-medium">Language</Label>

						<Select.Root
							onValueChange={(v) => onConfigChange(SETTINGS_KEYS.TTS_LANGUAGE, v)}
							type="single"
							value={languageSetting}
						>
							<Select.Trigger class="w-full">
								{languageSetting === 'auto' ? 'Automatic' : langName(languageSetting)}
							</Select.Trigger>

							<Select.Content>
								<Select.Item label="Automatic" value="auto">
									Automatic (the voice's language)
								</Select.Item>

								{#each engine.languages as l (l)}
									<Select.Item label={langName(l)} value={l}>{langName(l)}</Select.Item>
								{/each}
							</Select.Content>
						</Select.Root>
					</div>
				{/if}

				<div class="space-y-2 sm:col-span-2">
					<div class="flex items-center justify-between">
						<Label class="text-sm font-medium" for="tts-speed">Speed</Label>

						<span class="text-xs tabular-nums text-muted-foreground">{speed.toFixed(1)}×</span>
					</div>

					<input
						class="w-full accent-primary"
						id="tts-speed"
						max="1.8"
						min="0.6"
						oninput={(e) => onConfigChange(SETTINGS_KEYS.TTS_SPEED, Number(e.currentTarget.value))}
						step="0.1"
						type="range"
						value={speed}
					/>
				</div>
			</div>

			{#if engine}
				<p class="text-xs text-muted-foreground">
					{engine.label}: {engine.note}{engine.gpu
						? ` · ~${mb(engine.vram_mb)} VRAM while loaded`
						: ' · CPU'}
				</p>
			{/if}

			<div class="flex gap-2">
				<Input bind:value={testText} class="flex-1" placeholder="Text to try the voice with" />

				<Button disabled={!engine} onclick={test} variant="outline">
					{#if testing}
						<Square class="h-3.5 w-3.5" />
						Stop
					{:else}
						<Play class="h-3.5 w-3.5" />
						Test
					{/if}
				</Button>
			</div>
		</section>

		<!-- your voices -->
		<section class="space-y-4">
			<div class="space-y-1">
				<h3 class="text-sm font-semibold">Your voices</h3>

				<p class="text-xs text-muted-foreground">
					A voice is a {minSeconds}–{maxSeconds} s recording of one speaker (6–15 s of clear speech works
					best). Nothing is trained: the engines marked "clones voices" speak in it straight away. Stored
					in <code>{paths.voices ?? 'voices/'}</code>.
				</p>
			</div>

			{#if library.length}
				<ul class="divide-y divide-border/30 rounded-lg border border-border/40">
					{#each library as v (v.id)}
						<li class="space-y-1.5 px-3 py-2.5">
							<div class="flex items-center gap-2">
								<div class="min-w-0 flex-1">
									<div class="truncate text-sm font-medium">
										{v.name}
										{#if voiceSetting === v.id}
											<span class="ml-1 text-xs font-normal text-emerald-500">in use</span>
										{/if}
									</div>

									<div class="text-xs text-muted-foreground">
										{v.duration_s.toFixed(1)} s{v.language ? ` · ${langName(v.language)}` : ''}
										{v.transcript ? '' : ' · no transcript'}
									</div>
								</div>

								<Button
									onclick={() => playReference(v.id)}
									size="icon-sm"
									title="Play the reference clip"
									variant="ghost"
								>
									{#if previewing === v.id}<Square class="h-3.5 w-3.5" />{:else}<Play
											class="h-3.5 w-3.5"
										/>{/if}
								</Button>

								<Button
									onclick={() => {
										editing = editing === v.id ? null : v.id;
										editText = v.transcript;
									}}
									size="icon-sm"
									title="Edit the transcript"
									variant="ghost"
								>
									<Pencil class="h-3.5 w-3.5" />
								</Button>

								<Button
									disabled={!engine?.cloning}
									onclick={() => onConfigChange(SETTINGS_KEYS.TTS_VOICE, v.id)}
									size="sm"
									title={engine?.cloning
										? 'Read aloud with this voice'
										: 'Pick an engine that clones voices first'}
									variant="outline"
								>
									Use
								</Button>

								<Button
									onclick={() => removeVoice(v)}
									size="icon-sm"
									title="Delete"
									variant="ghost"
								>
									<Trash2 class="h-3.5 w-3.5" />
								</Button>
							</div>

							{#if editing === v.id}
								<div class="space-y-2">
									<Textarea bind:value={editText} class="min-h-[4rem] text-xs" />

									<div class="flex justify-end gap-2">
										<Button onclick={() => (editing = null)} size="sm" variant="ghost"
											>Cancel</Button
										>

										<Button onclick={() => saveTranscript(v)} size="sm">Save transcript</Button>
									</div>
								</div>
							{:else if v.transcript}
								<p class="line-clamp-2 text-xs text-muted-foreground">“{v.transcript}”</p>
							{/if}
						</li>
					{/each}
				</ul>
			{/if}

			<div class="space-y-3 rounded-lg border border-dashed border-border/60 p-3">
				<div class="text-sm font-medium">New voice</div>

				{#if recState === 'recording'}
					<div class="space-y-2 rounded-md bg-muted/40 p-3">
						<p class="text-xs text-muted-foreground">Read this aloud, in your normal voice:</p>

						<p class="text-sm leading-relaxed">{passage}</p>

						<div class="flex items-center gap-3 pt-1">
							<Mic class="h-4 w-4 shrink-0 text-destructive" />

							<div class="relative h-2 flex-1 overflow-hidden rounded-full bg-muted-foreground/20">
								<div
									class="absolute inset-y-0 left-0 rounded-full bg-primary transition-[width] duration-75"
									style="width: {Math.round(recLevel * 100)}%;"
								></div>
							</div>

							<span class="w-14 text-right text-xs tabular-nums text-muted-foreground">
								{recSeconds} / {maxSeconds} s
							</span>

							<Button onclick={stopRecording} size="sm" variant="destructive">
								<Square class="h-3.5 w-3.5" />
								Stop
							</Button>
						</div>
					</div>
				{:else}
					<div class="flex flex-wrap gap-2">
						<Button
							disabled={!isAudioRecordingSupported()}
							onclick={startRecording}
							size="sm"
							title={isAudioRecordingSupported()
								? 'Record from the microphone'
								: 'The microphone needs https (or localhost)'}
							variant="outline"
						>
							<Mic class="h-3.5 w-3.5" />
							{recState === 'ready' && clipName === 'recording.wav' ? 'Record again' : 'Record'}
						</Button>

						<Button onclick={() => fileInput?.click()} size="sm" variant="outline">
							<Upload class="h-3.5 w-3.5" />
							Upload a clip
						</Button>

						<input
							bind:this={fileInput}
							accept="audio/*,.wav,.mp3,.flac,.ogg,.webm,.m4a"
							class="hidden"
							onchange={pickFile}
							type="file"
						/>
					</div>
				{/if}

				{#if recState === 'ready' && clip}
					<div class="flex items-center gap-2">
						<audio class="h-8 flex-1" controls src={clipUrl}></audio>

						<Button onclick={discardClip} size="icon-sm" title="Discard" variant="ghost">
							<X class="h-3.5 w-3.5" />
						</Button>
					</div>

					<div class="grid gap-3 sm:grid-cols-[1fr_8rem]">
						<div class="space-y-1">
							<Label class="text-xs" for="voice-name">Name</Label>

							<Input bind:value={newName} id="voice-name" placeholder="e.g. my-voice" />
						</div>

						<div class="space-y-1">
							<Label class="text-xs" for="voice-lang">Language</Label>

							<Input bind:value={newLanguage} id="voice-lang" placeholder="en, fi, …" />
						</div>
					</div>

					<div class="space-y-1">
						<Label class="text-xs" for="voice-transcript">
							What is said in the clip (empty = transcribe it automatically)
						</Label>

						<Textarea
							bind:value={newTranscript}
							class="min-h-[4rem] text-xs"
							id="voice-transcript"
						/>
					</div>

					<div class="flex justify-end">
						<Button disabled={saving || !newName.trim()} onclick={saveVoice} size="sm">
							{#if saving}
								<LoaderCircle class="h-3.5 w-3.5 animate-spin" />
								Saving…
							{:else}
								<Check class="h-3.5 w-3.5" />
								Save voice
							{/if}
						</Button>
					</div>
				{/if}
			</div>
		</section>

		<!-- models -->
		<section class="space-y-4">
			<div class="flex items-center justify-between gap-3">
				<div class="space-y-1">
					<h3 class="text-sm font-semibold">Models</h3>

					<p class="text-xs text-muted-foreground">
						Download here, or copy the files into the folder shown and press Refresh.
					</p>
				</div>

				<Button
					disabled={unloading}
					onclick={unload}
					size="sm"
					title="Free the GPU memory speech models use"
					variant="outline"
				>
					<Power class="h-3.5 w-3.5" />
					Unload
				</Button>
			</div>

			{#if download?.running}
				<div class="space-y-1 rounded-lg border border-border/40 p-3">
					<div class="flex items-center justify-between gap-2 text-xs">
						<span class="truncate"
							>Downloading {download.engine} / {download.id}
							{download.file ? `· ${download.file}` : ''}</span
						>

						<span class="shrink-0 tabular-nums text-muted-foreground">
							{download.total ? `${mb(download.done / 1e6)} / ${mb(download.total / 1e6)}` : ''}
							{downloadPct.toFixed(0)}%
						</span>
					</div>

					<div class="h-1.5 w-full overflow-hidden rounded-full bg-muted">
						<div class="h-full bg-primary transition-all" style="width: {downloadPct}%;"></div>
					</div>

					<div class="flex justify-end">
						<Button class="h-6" onclick={cancelDownload} size="sm" variant="ghost">Cancel</Button>
					</div>
				</div>
			{/if}

			<div class="divide-y divide-border/30 rounded-lg border border-border/40">
				{#each catalog as c (c.engine)}
					{@const open = openEngines[c.engine] ?? c.engine === engine?.name}
					{@const have = c.items.filter((i) => i.installed).length}

					<div>
						<button
							class="flex w-full items-center justify-between gap-2 px-3 py-2.5 text-left text-sm hover:bg-muted/40"
							onclick={() => (openEngines[c.engine] = !open)}
							type="button"
						>
							<span class="font-medium">{c.label}</span>

							<span class="text-xs text-muted-foreground">
								{c.installed ? `${have} of ${c.items.length} on disk` : 'Python packages missing'}
							</span>
						</button>

						{#if open}
							<div class="space-y-2 px-3 pb-3">
								{#if !c.installed}
									<p class="flex items-start gap-1.5 text-xs text-amber-600 dark:text-amber-400">
										<CircleAlert class="mt-0.5 h-3.5 w-3.5 shrink-0" />
										{c.detail}
									</p>
								{/if}

								{#each c.items as item (item.id)}
									{@const busy =
										download?.running && download.engine === c.engine && download.id === item.id}

									<div class="flex items-center gap-2 text-sm">
										{#if item.installed}
											<Check class="h-4 w-4 shrink-0 text-emerald-500" />
										{:else}
											<Download class="h-4 w-4 shrink-0 text-muted-foreground" />
										{/if}

										<div class="min-w-0 flex-1">
											<div class="truncate">
												{item.label}
												{#if item.recommended}
													<span class="ml-1 text-xs text-muted-foreground">recommended</span>
												{/if}
											</div>

											<div class="truncate text-xs text-muted-foreground">
												{item.size_mb ? mb(item.size_mb) : ''}{item.note ? ` · ${item.note}` : ''}
												{item.installed ? '' : ` · from ${item.source}`}
											</div>
										</div>

										{#if !item.installed}
											<Button
												class="shrink-0"
												disabled={download?.running === true}
												onclick={() => startDownload(c.engine, item.id)}
												size="sm"
												variant="outline"
											>
												{#if busy}
													<LoaderCircle class="h-3.5 w-3.5 animate-spin" />
												{:else}
													<Download class="h-3.5 w-3.5" />
												{/if}
												Download
											</Button>
										{/if}
									</div>
								{/each}

								{#if paths[ENGINE_DIR[c.engine]]}
									<p class="flex items-center gap-1.5 pt-1 text-xs text-muted-foreground">
										<FolderOpen class="h-3.5 w-3.5 shrink-0" />

										<code class="truncate">{paths[ENGINE_DIR[c.engine]]}</code>
									</p>
								{/if}
							</div>
						{/if}
					</div>
				{/each}
			</div>

			<p class="text-xs text-muted-foreground">
				GPU engines load on first use and unload themselves after 10 minutes without speech, so the
				chat model keeps the card.
			</p>
		</section>
	{/if}
</div>
