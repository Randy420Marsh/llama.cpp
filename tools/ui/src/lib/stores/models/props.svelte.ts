/**
 * ModelPropsManager - Per-model props cache, modalities and thinking detection
 *
 * Owns the /props?model=<id> cache with TTL, the modality views over it,
 * and chat-template thinking detection. Created and owned by modelsStore;
 * the host owns the model lists that fetched modalities are mirrored onto.
 *
 * **API Inconsistency Workaround:**
 * In MODEL mode, `/props` returns modalities for the single model.
 * In ROUTER mode, `/props` has no modalities - must use `/props?model=<id>` per model.
 */

import {
	MODEL_PROPS_CACHE,
	REASONING_EFFORT_BASE_LEVELS,
	REASONING_EFFORT_BUDGET_LEVELS,
	REASONING_EFFORT_LABELS
} from '$lib/constants';
import { FileTypeCategory, ModelModality, ReasoningEffort } from '$lib/enums';
import { PropsService } from '$lib/services/props.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { serverStore } from '$lib/stores/server.svelte';
import type { ReasoningEffortLevel } from '$lib/types';
// deep imports, not the '$lib/utils' barrel: it re-exports modules that reach back
// into the stores, and going through it here would read a half-built module
import { TTLCache } from '$lib/utils/cache-ttl';
import { detectThinkingSupport } from '$lib/utils/chat-template-thinking-detector';
import { SvelteMap } from 'svelte/reactivity';

/**
 * The slice of modelsStore the manager reads. Kept narrow on purpose so it
 * cannot reach around the host's full surface; modelsStore implements this
 * structurally.
 */
export interface ModelPropsHost {
	/** Model rows the manager mirrors fetched modalities onto. */
	models: ModelOption[];
	readonly selectedModelName: string | null;
	readonly loadedModelIds: string[];
	isModelLoaded(modelId: string): boolean;
}

export class ModelPropsManager {
	/** Version counter for the cache - bumped on writes so $derived consumers recompute. */
	cacheVersion = $state(0);
	/**
	 * Model-specific props cache with TTL.
	 * Key: modelId, Value: props data including modalities.
	 */
	private cache = new TTLCache<string, ApiLlamaCppServerProps>({
		maxEntries: MODEL_PROPS_CACHE.MAX_ENTRIES,
		ttlMs: MODEL_PROPS_CACHE.TTL_MS
	});
	/** Props requests in flight, a second caller awaits the same request. */
	private fetching = new SvelteMap<string, Promise<ApiLlamaCppServerProps | null>>();

	/**
	 * Whether the selected model's chat template supports thinking/reasoning.
	 * Uses heuristic detection on the model's chat_template from /props.
	 *
	 * - MODEL mode: the global /props already describes the single loaded model,
	 *   so its chat_template is used directly and no per-model cache is involved
	 * - ROUTER mode: fetches /props?model=<id> for the selected model (cached),
	 *   triggering an async fetch if not yet cached
	 */
	get supportsThinking(): boolean {
		if (!serverStore.isRouterMode) {
			return detectThinkingSupport(serverStore.props?.chat_template ?? '');
		}

		const modelId = this.host.selectedModelName;

		if (!modelId) return false;

		if (!this.cache.get(modelId)) {
			this.fetchModelProps(modelId);
		}

		const props = this.getModelProps(modelId);

		return detectThinkingSupport(props?.chat_template ?? '');
	}

	/**
	 * The per-part `audio` field for the "Audio-capable models get" setting:
	 * 'transcript' asks for the Whisper transcript, 'native' for the model's own
	 * hearing (sent only to a model that has it: llama-server refuses it otherwise).
	 */
	audioRequestMode(modelId: string | null | undefined, choice: unknown): { audio?: string } {
		if (choice === 'transcript') return { audio: 'transcript' };

		if (choice === 'native' && this.hasAudioEncoder(modelId)) return { audio: 'native' };

		return {};
	}

	/** Map the router modalities, the only source available while a model is not loaded. */
	buildArchitectureModalities(
		architecture: ApiModelDataEntry['architecture']
	): ModelModalities | undefined {
		if (!architecture) return undefined;

		const inputs = architecture.input_modalities;

		return {
			audio: inputs.includes(FileTypeCategory.AUDIO),
			video: inputs.includes(FileTypeCategory.VIDEO),
			vision: inputs.includes(FileTypeCategory.IMAGE)
		};
	}

	/**
	 * Check if a specific model supports thinking.
	 * In MODEL mode the global /props describes the single loaded model.
	 * In ROUTER mode, fetches model props if not cached.
	 */
	checkModelSupportsThinking(modelId: string): boolean {
		if (!serverStore.isRouterMode) {
			return detectThinkingSupport(serverStore.props?.chat_template ?? '');
		}

		if (!modelId) return false;

		if (!this.cache.get(modelId)) {
			this.fetchModelProps(modelId);
		}

		const props = this.getModelProps(modelId);

		return detectThinkingSupport(props?.chat_template ?? '');
	}

	constructor(private host: ModelPropsHost) {}

	/** Fetch modalities for all loaded models from /props endpoint. */
	async fetchModalitiesForLoadedModels(): Promise<void> {
		const loadedModelIds = this.host.loadedModelIds;

		if (loadedModelIds.length === 0) return;

		const propsPromises = loadedModelIds.map((modelId) => this.fetchModelProps(modelId));

		try {
			const results = await Promise.all(propsPromises);

			this.host.models = this.host.models.map((model) => {
				const modelIndex = loadedModelIds.indexOf(model.model);

				if (modelIndex === -1) return model;

				const props = results[modelIndex];

				if (!props?.modalities) return model;

				return { ...model, modalities: this.buildModalities(props.modalities) };
			});

			this.cacheVersion++;
		} catch (error) {
			console.warn('Failed to fetch modalities for loaded models:', error);
		}
	}

	/**
	 * Fetch props for a specific model from /props endpoint.
	 * Uses caching to avoid redundant requests.
	 *
	 * In ROUTER mode, this only fetches props if the model is loaded,
	 * since unloaded models return 400 from /props endpoint.
	 *
	 * @param modelId - Model identifier to fetch props for
	 * @returns Props data or null if fetch failed or model not loaded
	 */
	async fetchModelProps(modelId: string): Promise<ApiLlamaCppServerProps | null> {
		const cached = this.cache.get(modelId);

		if (cached) return cached;

		if (serverStore.isRouterMode && !this.host.isModelLoaded(modelId)) {
			return null;
		}

		const inFlight = this.fetching.get(modelId);

		if (inFlight) return inFlight;

		const request = PropsService.fetchForModel(modelId)
			.then((props) => {
				this.cache.set(modelId, props);
				this.cacheVersion++;

				return props;
			})
			.catch((error) => {
				console.warn(`Failed to fetch props for model ${modelId}:`, error);

				return null;
			})
			.finally(() => this.fetching.delete(modelId));

		this.fetching.set(modelId, request);

		return request;
	}

	/**
	 * What audio becomes on the server by default (/props `audio_input`): native
	 * (the model hears it), transcript, both or none; '' when unknown.
	 */
	getAudioInput(modelId?: string | null): string {
		return this.audioProps(modelId)?.audio_input ?? '';
	}

	getModelContextSize(modelId: string): number | null {
		const props = this.getModelProps(modelId);
		const nCtx = props?.default_generation_settings?.n_ctx;

		return typeof nCtx === 'number' ? nCtx : null;
	}

	getModelModalities(modelId: string): ModelModalities | null {
		if (!serverStore.isRouterMode && serverStore.props?.modalities) {
			return this.buildModalities(serverStore.props.modalities);
		}

		const model = this.host.models.find((m) => m.model === modelId || m.id === modelId);

		if (model?.modalities) {
			return model.modalities;
		}

		const props = this.cache.get(modelId);

		if (props?.modalities) {
			return this.buildModalities(props.modalities);
		}

		return null;
	}

	getModelModalitiesArray(modelId: string): ModelModality[] {
		const modalities = this.getModelModalities(modelId);

		if (!modalities) return [];

		const result: ModelModality[] = [];

		if (modalities.vision) result.push(ModelModality.VISION);

		if (modalities.audio) result.push(ModelModality.AUDIO);

		if (modalities.video) result.push(ModelModality.VIDEO);

		return result;
	}

	getModelProps(modelId: string): ApiLlamaCppServerProps | null {
		return this.cache.get(modelId);
	}

	/** The level a request without reasoning_effort runs at ('' when unknown). */
	getReasoningEffortDefault(modelId?: string | null): string {
		void this.cacheVersion;

		const props = !serverStore.isRouterMode
			? serverStore.props
			: modelId
				? this.getModelProps(modelId)
				: null;

		return props?.reasoning_effort_default ?? '';
	}

	/**
	 * Reasoning effort levels the model's chat template honors, from lowest to highest.
	 * Empty when the template honors none, the menu then maps its levels to token budgets.
	 * In ROUTER mode, fetches model props if not cached.
	 */
	getReasoningEfforts(modelId: string): ReasoningEffort[] {
		let props: ApiLlamaCppServerProps | null = serverStore.props;

		if (serverStore.isRouterMode) {
			if (!modelId) return [];

			if (!this.cache.get(modelId)) {
				this.fetchModelProps(modelId);
			}

			props = this.getModelProps(modelId);
		}

		const levels: string[] = props?.chat_template_caps?.reasoning_effort_levels ?? [];

		return levels.filter((level): level is ReasoningEffort => level in REASONING_EFFORT_LABELS);
	}

	/** Reasoning menu of the model: the template levels, or the token budget levels. */
	getReasoningLevels(modelId: string): ReasoningEffortLevel[] {
		const efforts = this.getReasoningEfforts(modelId);

		if (efforts.length === 0) return REASONING_EFFORT_BUDGET_LEVELS;

		return [
			...REASONING_EFFORT_BASE_LEVELS,
			...efforts.map((value) => ({ label: REASONING_EFFORT_LABELS[value], value }))
		];
	}

	/** llama-server transcribes with its --asr-url whisper (POST /asr/transcribe); null = unknown */
	hasAsr(modelId?: string | null): boolean | null {
		return this.audioProps(modelId)?.asr ?? null;
	}

	/**
	 * The model hears audio by itself (an audio encoder, /props `audio_encoder`);
	 * a server without that field: it takes audio and does not transcribe it.
	 */
	hasAudioEncoder(modelId?: string | null): boolean {
		if (!modelId || !this.modelSupportsAudio(modelId)) return false;

		const props = this.audioProps(modelId);

		if (typeof props?.audio_encoder === 'boolean') return props.audio_encoder;

		const input = props?.audio_input ?? '';

		return input === '' || input === 'native' || input === 'both';
	}

	isModelPropsFetching(modelId: string): boolean {
		return this.fetching.has(modelId);
	}

	modelSupportsAudio(modelId: string): boolean {
		return this.getModelModalities(modelId)?.audio ?? false;
	}

	modelSupportsVideo(modelId: string): boolean {
		return this.getModelModalities(modelId)?.video ?? false;
	}

	modelSupportsVision(modelId: string): boolean {
		return this.getModelModalities(modelId)?.vision ?? false;
	}

	/** The effort as the model's menu offers it, DEFAULT when the menu lacks it. */
	resolveReasoningEffort(modelId: string, effort: ReasoningEffort): ReasoningEffort {
		return this.getReasoningLevels(modelId).some((level) => level.value === effort)
			? effort
			: ReasoningEffort.DEFAULT;
	}

	/**
	 * Update modalities for a specific model.
	 * Called when a model is loaded or when we need fresh modality data.
	 */
	async updateModelModalities(modelId: string): Promise<void> {
		const props = await this.fetchModelProps(modelId);

		if (!props?.modalities) return;

		this.host.models = this.host.models.map((model) =>
			model.model === modelId
				? { ...model, modalities: this.buildModalities(props.modalities!) }
				: model
		);

		this.cacheVersion++;
	}

	private audioProps(modelId?: string | null) {
		void this.cacheVersion;

		return !serverStore.isRouterMode
			? serverStore.props
			: modelId
				? this.getModelProps(modelId)
				: null;
	}

	private buildModalities(
		modalities: NonNullable<ApiLlamaCppServerProps['modalities']>
	): ModelModalities {
		return {
			audio: modalities.audio ?? false,
			video: modalities.video ?? false,
			vision: modalities.vision ?? false
		};
	}
}
