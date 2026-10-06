<script lang="ts">
	import { Checkbox } from '$lib/components/ui/checkbox';

	interface Props {
		wrapLines: boolean;
		fontSize: number;
		onWrapChange: (value: boolean) => void;
		/** Live preview while dragging the slider. */
		onFontInput: (value: number) => void;
		/** Persisted when the slider is released. */
		onFontChange: (value: number) => void;
	}

	let { fontSize, onFontChange, onFontInput, onWrapChange, wrapLines }: Props = $props();

	function handleFontInput(e: Event) {
		onFontInput(Number((e.target as HTMLInputElement).value));
	}

	function handleFontChange(e: Event) {
		onFontChange(Number((e.target as HTMLInputElement).value));
	}
</script>

<div class="mb-2 flex w-full items-center justify-end gap-4">
	<label class="flex items-center gap-1.5 text-xs text-muted-foreground">
		<Checkbox
			checked={wrapLines}
			id="preview-wrap-lines"
			onCheckedChange={(checked) => onWrapChange(checked === true)}
		/>
		Wrap lines
	</label>

	<label class="flex items-center gap-1.5 text-xs text-muted-foreground" for="preview-font-size">
		<span>Font</span>

		<input
			class="h-1.5 w-28 cursor-pointer accent-primary"
			id="preview-font-size"
			max={24}
			min={10}
			onchange={handleFontChange}
			oninput={handleFontInput}
			step={1}
			type="range"
			value={fontSize}
		/>

		<span class="w-8 text-right tabular-nums">{fontSize}px</span>
	</label>
</div>
