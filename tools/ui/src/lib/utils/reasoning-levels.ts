/**
 * Reasoning effort levels of the loaded model.
 *
 * The server (this fork) reports in /props which `reasoning_effort` values the
 * model's chat template accepts (`reasoning_efforts`, e.g. low / medium / xhigh
 * for Qwen3.8) and which one it uses when a request names none
 * (`reasoning_effort_default`). A model that reports levels gets exactly those
 * in the reasoning menu, sent as `reasoning_effort`; the generic Low / Medium /
 * High / Max entries are thinking-token budgets and are only used for models
 * without levels. Sending a level the template does not know makes it raise.
 */

/** The server's probe order, lowest first. */
export const EFFORT_LADDER = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];

const EFFORT_LABELS: Record<string, string> = {
	high: 'High',
	low: 'Low',
	max: 'Max',
	medium: 'Medium',
	minimal: 'Minimal',
	none: 'Off',
	xhigh: 'Extra high'
};

export function effortLabel(level: string): string {
	return EFFORT_LABELS[level] ?? (level ? level.charAt(0).toUpperCase() + level.slice(1) : '');
}

function rank(level: string): number {
	const i = EFFORT_LADDER.indexOf(level);

	// unknown names (e.g. a budget preset) rank above everything: they mean "a lot"
	return i < 0 ? EFFORT_LADDER.length : i;
}

/**
 * The model level closest to `wanted` on the ladder (ties go to the higher one),
 * so a saved "max" or "high" still means "as much as this model offers".
 * Returns null when the model reports no levels.
 */
export function nearestEffort(wanted: string, supported: string[]): string | null {
	const levels = supported.filter((l) => l !== 'none');

	if (levels.length === 0) return null;

	if (levels.includes(wanted)) return wanted;

	const w = rank(wanted);

	let best = levels[0];

	for (const l of levels) {
		const d = Math.abs(rank(l) - w);
		const bestD = Math.abs(rank(best) - w);

		if (d < bestD || (d === bestD && rank(l) > rank(best))) best = l;
	}

	return best;
}

/**
 * What a reply was sent with, stored on the message as `reasoningUsed`:
 *   "level:xhigh"     the model's own effort level
 *   "default:xhigh"   no level sent; the template's default (when the server knows it)
 *   "default"         no level sent, default unknown
 *   "budget:8192"     a thinking-token budget (-1 = unlimited)
 *   "off"             thinking disabled
 */
export function describeReasoningUsed(used: string | undefined): string {
	if (!used) return '';

	const [kind, value] = used.split(':', 2);

	switch (kind) {
		case 'level':
			return `Effort: ${effortLabel(value)}`;
		case 'default':
			return value ? `Effort: ${effortLabel(value)} (model default)` : 'Effort: model default';
		case 'budget':
			return Number(value) < 0
				? 'Thinking: no budget'
				: `Thinking budget: ${Number(value).toLocaleString()} tokens`;
		case 'off':
			return 'Thinking off';
		default:
			return '';
	}
}
