<script lang="ts">
	import ChatAttachmentsPreviewTextToolbar from './ChatAttachmentsPreviewTextToolbar.svelte';
	import { SyntaxHighlightedCode } from '$lib/components/app';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { settingsStore } from '$lib/stores';

	interface Props {
		displayTextContent: string | undefined;
		language: string;
	}

	let { displayTextContent, language }: Props = $props();

	// Preview text options (Settings -> Display), initialised from persisted values.
	let wrapLines = $state(Boolean(settingsStore.config[SETTINGS_KEYS.PREVIEW_WRAP_LINES]));
	const initialFontSize = (() => {
		const value = Number(settingsStore.config[SETTINGS_KEYS.PREVIEW_FONT_SIZE]);

		return Number.isFinite(value) ? Math.min(24, Math.max(10, Math.round(value))) : 14;
	})();
	let fontSize = $state(initialFontSize);
</script>

{#if displayTextContent}
	<div class="w-full px-4 pb-4">
		<ChatAttachmentsPreviewTextToolbar
			{fontSize}
			onFontChange={(value) => {
				fontSize = value;
				settingsStore.updateConfig(SETTINGS_KEYS.PREVIEW_FONT_SIZE, value);
			}}
			onFontInput={(value) => (fontSize = value)}
			onWrapChange={(value) => {
				wrapLines = value;
				settingsStore.updateConfig(SETTINGS_KEYS.PREVIEW_WRAP_LINES, value);
			}}
			{wrapLines}
		/>

		<SyntaxHighlightedCode
			code={displayTextContent}
			{fontSize}
			{language}
			maxHeight="none"
			wrap={wrapLines}
		/>
	</div>
{/if}
