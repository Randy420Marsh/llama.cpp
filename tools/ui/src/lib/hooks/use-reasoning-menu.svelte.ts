import {
	REASONING_EFFORT_LABELS,
	REASONING_EFFORT_LEVELS,
	REASONING_EFFORT_TOKENS
} from '$lib/constants';
import { ReasoningEffort } from '$lib/enums';
import { conversationsStore, modelsStore, serverStore } from '$lib/stores';
import type { ReasoningEffortLevel } from '$lib/types';
import type { DatabaseMessage } from '$lib/types/database';
import { getConversationModel } from '$lib/utils';
import { effortLabel, nearestEffort } from '$lib/utils/reasoning-levels';

export interface UseReasoningMenuReturn {
	readonly modelSupportsThinking: boolean;
	readonly thinkingEnabled: boolean;
	readonly isReasoningActive: boolean;
	readonly isOff: boolean;
	readonly currentEffort: ReasoningEffort;
	/** What the trigger shows: the level that will really be sent */
	readonly currentLabel: string;
	readonly levels: ReasoningEffortLevel[];
	isSelected(level: ReasoningEffortLevel): boolean;
	tokenLabel(level: ReasoningEffortLevel): string | null;
	select(level: ReasoningEffortLevel): void;
}

/**
 * Shared reactive state and helpers for the reasoning effort menu.
 *
 * Used by both the desktop dropdown (`ChatFormActionAddReasoningSubmenu`)
 * and the mobile sheet (`ChatFormActionAddSheet`) to avoid duplicating the
 * thinking-support derivation and the effort selection logic.
 *
 * A model whose server reports effort levels (/props `reasoning_efforts`) gets
 * exactly those, sent as `reasoning_effort` (see utils/reasoning-levels.ts).
 * Other models get the generic levels, which are thinking-token budgets.
 */
export function useReasoningMenu(): UseReasoningMenuReturn {
	const conversationModel = $derived(
		getConversationModel(conversationsStore.activeMessages as DatabaseMessage[])
	);
	const propsModelId = $derived(
		serverStore.isRouterMode ? modelsStore.selectedModelName || conversationModel : null
	);
	// a router chat can carry reasoning from an earlier turn before the props
	// cache is primed, so a model that already produced thinking still qualifies
	const modelSupportsThinkingFromMessages = $derived.by(() => {
		const modelId = serverStore.isRouterMode
			? modelsStore.selectedModelName || conversationModel
			: null;

		if (!modelId) return false;

		return conversationsStore.activeMessages.some(
			(m) => m.role === 'assistant' && m.model === modelId && !!m.reasoningContent
		);
	});
	const modelSupportsThinking = $derived.by(() => {
		void modelsStore.loadedModelIds;
		void modelsStore.props.cacheVersion;

		if (serverStore.isRouterMode) {
			const modelId = modelsStore.selectedModelName || conversationModel;

			return (
				modelsStore.props.checkModelSupportsThinking(modelId ?? '') ||
				modelSupportsThinkingFromMessages
			);
		}

		return modelsStore.props.supportsThinking || modelSupportsThinkingFromMessages;
	});
	const modelEfforts = $derived.by(() => {
		void serverStore.props;

		return modelsStore.props.getReasoningEfforts(propsModelId);
	});
	const effortDefault = $derived.by(() => {
		void serverStore.props;

		return modelsStore.props.getReasoningEffortDefault(propsModelId);
	});
	const currentEffort = $derived(conversationsStore.preferences.getReasoningEffort());
	const thinkingEnabled = $derived(
		currentEffort !== ReasoningEffort.OFF && currentEffort !== ReasoningEffort.DEFAULT
	);
	// a saved choice this model does not have ("max", "high") is shown -- and
	// sent -- as its nearest real level, never silently as something else
	const effectiveEffort = $derived(
		thinkingEnabled && modelEfforts.length > 0
			? (nearestEffort(currentEffort, modelEfforts) ?? currentEffort)
			: currentEffort
	);
	const levels = $derived<ReasoningEffortLevel[]>(
		modelEfforts.length > 0
			? [
					{ label: 'Default', value: ReasoningEffort.DEFAULT },
					{ label: 'Off', value: ReasoningEffort.OFF },
					...modelEfforts.map((level) => ({ label: effortLabel(level), value: level }))
				]
			: REASONING_EFFORT_LEVELS
	);
	// Thinking is effectively on (lightbulb lit) either when an explicit effort
	// is selected, or when the effort is left at "Default" and the model
	// supports thinking.
	const isReasoningActive = $derived(
		thinkingEnabled || (currentEffort === ReasoningEffort.DEFAULT && modelSupportsThinking)
	);

	return {
		get currentEffort() {
			return currentEffort;
		},
		get currentLabel() {
			if (currentEffort === ReasoningEffort.DEFAULT && effortDefault) {
				return `Default (${effortLabel(effortDefault)})`;
			}

			return modelEfforts.length > 0 || !REASONING_EFFORT_LABELS[effectiveEffort]
				? effortLabel(effectiveEffort)
				: REASONING_EFFORT_LABELS[effectiveEffort];
		},
		get isOff() {
			return currentEffort === ReasoningEffort.OFF;
		},
		get isReasoningActive() {
			return isReasoningActive;
		},
		isSelected(level: ReasoningEffortLevel): boolean {
			return effectiveEffort === level.value;
		},
		get levels() {
			return levels;
		},
		get modelSupportsThinking() {
			return modelSupportsThinking;
		},
		select(level: ReasoningEffortLevel): void {
			conversationsStore.preferences.setReasoningEffort(level.value as ReasoningEffort);
		},
		get thinkingEnabled() {
			return thinkingEnabled;
		},
		tokenLabel(level: ReasoningEffortLevel): string | null {
			if (level.value === ReasoningEffort.DEFAULT) {
				return effortDefault ? `Model default: ${effortLabel(effortDefault)}` : 'Model default';
			}

			if (modelEfforts.length > 0) {
				// the model's own levels: no token cap, the template decides
				return level.value === ReasoningEffort.OFF ? null : 'Model level';
			}

			const tokens = REASONING_EFFORT_TOKENS[level.value];

			if (tokens === undefined) return null;

			return tokens === -1 ? 'Unlimited' : `Max ${tokens.toLocaleString()} tokens`;
		}
	};
}
