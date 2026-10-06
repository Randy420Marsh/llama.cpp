<script lang="ts">
	import { Mic, Volume2 } from '@lucide/svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { settingsStore } from '$lib/stores';
	import type { AudioRecorder } from '$lib/utils/browser-only';

	/** Fixed silence-timeout choices (ms) offered as a compact select. */
	const SILENCE_OPTIONS = [1000, 2000, 3000, 5000, 8000, 12000];

	let {
		audioRecorder,
		clip = false,
		level = 0,
		peak = 0
	}: {
		/** Live input level 0..1 from the meter callback. */
		level?: number;
		/** Peak-hold level 0..1. */
		peak?: number;
		/** True while the peak is at full scale. */
		clip?: boolean;
		/** The in-flight recorder, for live gain / noise-cancelling tweaks. */
		audioRecorder?: AudioRecorder;
	} = $props();

	let config = $derived(settingsStore.config);

	// The config is a string-keyed index signature, so coerce for typed use.
	let micGain = $derived(Number(config[SETTINGS_KEYS.MIC_GAIN] ?? 1) || 0);
	let noiseCancelling = $derived(Boolean(config[SETTINGS_KEYS.MIC_NOISE_CANCELLING]));
	let autoStop = $derived(Boolean(config[SETTINGS_KEYS.MIC_AUTO_STOP]));
	let autoSend = $derived(Boolean(config[SETTINGS_KEYS.MIC_AUTO_SEND]));
	let silenceMs = $derived(
		(() => {
			const n = Number(config[SETTINGS_KEYS.MIC_AUTO_STOP_SILENCE_MS]);

			return SILENCE_OPTIONS.includes(n) ? n : 5000;
		})()
	);

	// Live gain tweak on the in-flight recording; the setting itself is committed
	// on `change` so a half-drag does not spam localStorage.
	function handleGainInput(e: Event) {
		audioRecorder?.setGain(Number((e.target as HTMLInputElement).value));
	}

	function handleGainChange(e: Event) {
		settingsStore.updateConfig(
			SETTINGS_KEYS.MIC_GAIN,
			Number((e.target as HTMLInputElement).value)
		);
	}

	function handleSilenceChange(e: Event) {
		settingsStore.updateConfig(
			SETTINGS_KEYS.MIC_AUTO_STOP_SILENCE_MS,
			Number((e.target as HTMLSelectElement).value)
		);
	}
</script>

<div class="mx-3 mb-2 rounded-xl border bg-muted/40 px-3 py-2">
	<div class="flex items-center gap-3">
		<!-- Level meter: live fill + peak-hold marker + clip flag -->
		<div class="flex items-center gap-2">
			<Mic class="h-4 w-4 shrink-0 text-muted-foreground" />

			<div class="relative h-2 w-24 overflow-hidden rounded-full bg-muted-foreground/20">
				<div
					class:bg-destructive={clip}
					class:bg-primary={!clip}
					class="absolute inset-y-0 left-0 rounded-full transition-[width] duration-75"
					style="width: {Math.round(level * 100)}%;"
				></div>

				<div
					class="absolute inset-y-0 w-0.5 bg-foreground/70"
					style="left: calc({Math.round(peak * 100)}% - 1px);"
				></div>
			</div>

			<span class="w-8 text-[10px] tabular-nums text-muted-foreground">
				{clip ? 'CLIP' : `${Math.round(level * 100)}%`}
			</span>
		</div>

		<!-- Live gain + capture toggles -->
		<div class="flex min-w-0 flex-1 flex-wrap items-center gap-x-4 gap-y-2">
			<label class="flex items-center gap-2" for="mic-gain-slider">
				<span class="text-xs text-muted-foreground">Gain</span>

				<input
					class="h-1.5 w-28 cursor-pointer accent-primary"
					id="mic-gain-slider"
					max={3}
					min={0}
					onchange={handleGainChange}
					oninput={handleGainInput}
					step={0.05}
					type="range"
					value={micGain}
				/>

				<span class="w-10 text-right text-xs tabular-nums text-muted-foreground">
					{micGain.toFixed(2)}&times;
				</span>
			</label>

			<label class="flex items-center gap-1.5 text-xs text-muted-foreground">
				<Checkbox
					checked={noiseCancelling}
					id="mic-noise-cancelling"
					onCheckedChange={(checked) => {
						settingsStore.updateConfig(SETTINGS_KEYS.MIC_NOISE_CANCELLING, checked === true);
						void audioRecorder?.setNoiseCancelling(checked === true);
					}}
				/>
				Noise cancel
			</label>

			<label class="flex items-center gap-1.5 text-xs text-muted-foreground">
				<Checkbox
					checked={autoStop}
					id="mic-auto-stop"
					onCheckedChange={(checked) =>
						settingsStore.updateConfig(SETTINGS_KEYS.MIC_AUTO_STOP, checked === true)}
				/>
				Auto-stop
			</label>

			{#if autoStop}
				<label class="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
					<span>after</span>

					<select
						class="h-6 min-w-10 shrink-0 rounded border bg-background px-1.5 text-right text-xs tabular-nums"
						onchange={handleSilenceChange}
						value={silenceMs}
					>
						{#each SILENCE_OPTIONS as ms (ms)}
							<option value={ms}>
								{ms / 1000}s
							</option>
						{/each}
					</select>

					<span>silence</span>
				</label>
			{/if}

			<label class="flex items-center gap-1.5 text-xs text-muted-foreground">
				<Checkbox
					checked={autoSend}
					id="mic-auto-send"
					onCheckedChange={(checked) =>
						settingsStore.updateConfig(SETTINGS_KEYS.MIC_AUTO_SEND, checked === true)}
				/>

				<Volume2 class="h-3 w-3" />
				Auto-send
			</label>
		</div>
	</div>
</div>
