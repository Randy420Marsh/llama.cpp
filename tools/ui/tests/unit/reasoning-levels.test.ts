import { describeReasoningUsed, effortLabel, nearestEffort } from '$lib/utils/reasoning-levels';
import { describe, expect, it } from 'vitest';

// Qwen3.8's template accepts exactly these and raises on anything else
const QWEN38 = ['low', 'medium', 'xhigh'];

describe('nearestEffort', () => {
	it('keeps a level the model has', () => {
		expect(nearestEffort('medium', QWEN38)).toBe('medium');
		expect(nearestEffort('xhigh', QWEN38)).toBe('xhigh');
	});

	it('maps a saved generic choice to the closest real level, ties going up', () => {
		expect(nearestEffort('max', QWEN38)).toBe('xhigh');
		expect(nearestEffort('high', QWEN38)).toBe('xhigh');
		expect(nearestEffort('minimal', QWEN38)).toBe('low');
	});

	it('treats an unknown name as "a lot"', () => {
		expect(nearestEffort('ultra', QWEN38)).toBe('xhigh');
	});

	it('returns null when the model has no levels', () => {
		expect(nearestEffort('high', [])).toBeNull();
		expect(nearestEffort('high', ['none'])).toBeNull();
	});
});

describe('effortLabel', () => {
	it('names the ladder levels', () => {
		expect(effortLabel('xhigh')).toBe('Extra high');
		expect(effortLabel('low')).toBe('Low');
		expect(effortLabel('turbo')).toBe('Turbo');
	});
});

describe('describeReasoningUsed', () => {
	it('says what a reply was requested with', () => {
		expect(describeReasoningUsed('level:xhigh')).toBe('Effort: Extra high');
		expect(describeReasoningUsed('default:medium')).toBe('Effort: Medium (model default)');
		expect(describeReasoningUsed('default')).toBe('Effort: model default');
		expect(describeReasoningUsed('budget:8192')).toBe(
			`Thinking budget: ${(8192).toLocaleString()} tokens`
		);
		expect(describeReasoningUsed('budget:-1')).toBe('Thinking: no budget');
		expect(describeReasoningUsed('off')).toBe('Thinking off');
		expect(describeReasoningUsed(undefined)).toBe('');
	});
});
