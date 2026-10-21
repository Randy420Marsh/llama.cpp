import { toStandaloneSvg } from '$lib/utils';
import { describe, expect, it } from 'vitest';

// A representative in-DOM serialization: Mermaid edge labels inside a
// foreignObject carry HTML void tags (<br>, <img>) that the DOM leaves
// unclosed. Invalid in XML, which is how a downloaded .svg is parsed.
const LIVE_SVG =
	'<svg id="mermaid-1" xmlns="http://www.w3.org/2000/svg" width="100" height="100">' +
	'<g><foreignObject width="276" height="28"><div style="white-space: nowrap;" ' +
	'xmlns="http://www.w3.org/1999/xhtml"><span class="nodeLabel"><p>a<br>b<br/>c</p></span></div></foreignObject></g>' +
	'<img src="data:image/png;base64,x" width="16" height="16"/>' +
	'</svg>';

describe('toStandaloneSvg', () => {
	it('self-closes HTML void tags', () => {
		const out = toStandaloneSvg(LIVE_SVG);
		expect(out).toContain('a<br/>b<br/>c');
		expect(out).not.toContain('<br>');
	});

	it('keeps already self-closed void tags untouched', () => {
		const out = toStandaloneSvg('<svg xmlns="http://www.w3.org/2000/svg"><img src="x.png" width="1" height="1"/></svg>');
		expect(out).toContain('<img src="x.png" width="1" height="1"/>');
		expect(out).not.toContain('<img src="x.png" width="1" height="1//>');
	});

	it('leaves normal elements closed with end tags', () => {
		const out = toStandaloneSvg(LIVE_SVG);
		expect(out).toContain('</span>');
		expect(out).toContain('</svg>');
	});

	it('prepends the XML declaration when absent', () => {
		expect(toStandaloneSvg(LIVE_SVG)).toMatch(/^<\?xml version="1.0" encoding="UTF-8"\?>\n/);
	});

	it('keeps an existing declaration and does not duplicate it', () => {
		const once = toStandaloneSvg('<?xml version="1.0" encoding="UTF-8"?>\n' + LIVE_SVG);
		expect(once).toContain(LIVE_SVG.replace('<br>', '<br/>'));
		const twice = toStandaloneSvg(once);
		expect(twice).toBe(once);
	});

	it('returns empty string for blank input', () => {
		expect(toStandaloneSvg('')).toBe('');
		expect(toStandaloneSvg('  \n ')).toBe('');
	});
});
