import fs from "node:fs/promises";
import path from "node:path";
import {
  SpreadsheetFile,
  Workbook,
} from "file:///C:/Users/xsp/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = "C:\\Users\\xsp\\Desktop\\DRL代码\\paper_delivery\\EAAI_fulltext_rewrite_v2_2026-08-09";
const SOURCE = path.join(ROOT, "evidence", "evidence_architecture_source_v2.json");
const OUTPUT = path.join(ROOT, "evidence", "Evidence_Architecture_v2.xlsx");
const PREVIEW_DIR = path.join(ROOT, "qa", "evidence_workbook_previews");
const QA_OUTPUT = path.join(ROOT, "qa", "evidence_workbook_inspection.json");
const source = JSON.parse(await fs.readFile(SOURCE, "utf8"));
await fs.mkdir(PREVIEW_DIR, { recursive: true });

const wb = Workbook.create();
const NAVY = "#1D3D63";
const BLUE = "#337FB8";
const PALE = "#DCECF7";
const GREEN = "#E2F0D9";
const AMBER = "#FFF2CC";
const RED = "#FCE4D6";

function title(sheet, text, subtitle, lastCol) {
  sheet.mergeCells(`A1:${lastCol}1`);
  sheet.getRange("A1").values = [[text]];
  sheet.getRange("A1").format = { fill: NAVY, font: { bold: true, color: "#FFFFFF", size: 16 }, rowHeight: 32, verticalAlignment: "center" };
  sheet.mergeCells(`A2:${lastCol}2`);
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange("A2").format = { fill: PALE, font: { italic: true, color: "#405264" }, rowHeight: 44, wrapText: true, verticalAlignment: "center" };
}

function addTable(sheetName, heading, subtitle, headers, data, widths, lastCol) {
  const sheet = wb.worksheets.add(sheetName);
  title(sheet, heading, subtitle, lastCol);
  const range = sheet.getRangeByIndexes(3, 0, data.length + 1, headers.length);
  range.values = [headers, ...data];
  sheet.tables.add(range, true, `${sheetName.replaceAll(" ", "").replaceAll("-", "")}Table`);
  sheet.freezePanes.freezeRows(4);
  sheet.getRangeByIndexes(4, 0, Math.max(data.length, 1), headers.length).format = { wrapText: true, verticalAlignment: "top", rowHeight: 60 };
  headers.forEach((_, i) => { sheet.getRangeByIndexes(0, i, data.length + 4, 1).format.columnWidth = widths[i]; });
  return sheet;
}

const factData = source.facts.map(f => [f.fact_id, f.category, f.statement, f.value, f.unit, f.source, f.locator, f.strength, f.boundary]);
const fact = addTable("Fact Map", "Fact Map v2", "Every number is anchored to frozen v3.2.14 evidence; the stated boundary controls manuscript wording.",
  ["Fact ID", "Category", "Statement", "Value", "Unit", "Source", "Locator", "Strength", "Boundary"], factData,
  [10, 18, 40, 33, 22, 42, 48, 23, 40], "I");
fact.getRange(`A5:A${factData.length + 4}`).format = { fill: GREEN, font: { bold: true } };

const claimData = source.claims.map(c => [c.claim_id, c.section, c.claim, c.support, c.max_strength, c.forbidden]);
const claim = addTable("Claim Map", "Claim Map v2", "Claims are permissions, not prose templates. Non-significance is never promoted to equivalence.",
  ["Claim ID", "Section", "Permitted claim", "Support", "Maximum strength", "Forbidden inference"], claimData,
  [10, 16, 68, 36, 28, 48], "F");
claim.getRange(`A5:A${claimData.length + 4}`).format = { fill: GREEN, font: { bold: true } };
claim.getRange(`F5:F${claimData.length + 4}`).format = { fill: RED, wrapText: true };

const evidenceData = source.claims.map(c => {
  const factIds = [...c.support.matchAll(/F\d{2}/g)].map(m => m[0]);
  const matched = source.facts.filter(f => factIds.includes(f.fact_id));
  const locations = matched.map(f => `${f.fact_id}: ${f.source} (${f.locator})`).join("\n");
  const evidenceClass = matched.length ? "A: frozen internal evidence" : (c.section === "Methods" ? "A: protocol/implementation" : "A/D: synthesis");
  return [c.claim_id, c.section, c.claim, c.support, evidenceClass, locations || "See protocol/implementation or verified D-class claim map", c.max_strength, c.forbidden, "PASS at architecture gate; recheck exact final sentence"];
});
const evidence = addTable("Evidence Matrix", "Evidence Matrix v2", "The exact manuscript sentence must be rechecked against its evidence before final citation and audit.",
  ["Claim ID", "Section", "Claim", "Support IDs", "Evidence class", "Exact source locations", "Maximum strength", "Forbidden inference", "Gate"], evidenceData,
  [10, 15, 60, 30, 25, 62, 28, 45, 35], "I");
evidence.getRange(`I5:I${evidenceData.length + 4}`).format = { fill: GREEN, wrapText: true };

const figData = source.figure_storyline.map(f => [f.unit, f.assets, f.purpose, f.claim_ids, f.placement, f.boundary]);
const figures = addTable("Figure Storyline", "Eight evidence units", "M01-M10 are the frozen third-round default assets; V figures are non-inferential displays.",
  ["Unit", "Assets", "Purpose", "Claim IDs", "Placement", "Boundary"], figData,
  [10, 40, 40, 24, 20, 46], "F");
figures.getRange(`A5:A${figData.length + 4}`).format = { fill: GREEN, font: { bold: true } };

const glossaryData = source.glossary.map(r => [r.term, r.definition, r.control]);
addTable("Terminology", "Terminology and abbreviations", "Use these labels consistently across manuscript, figures, captions, supplement and source data.",
  ["Term", "Definition", "Usage control"], glossaryData, [30, 55, 55], "C");

const paraData = source.paragraph_plan.map(r => [r.section, r.paragraphs, r.function, r.claims, r.evidence]);
addTable("Paragraph Plan", "Paragraph plan v2", "Writing order remains Methods, Results, Discussion, Introduction, Conclusions, Abstract, Title, Keywords and Highlights.",
  ["Section", "Paragraphs", "Scientific function", "Claims", "Evidence"], paraData, [12, 18, 62, 28, 52], "E");

const authorData = source.author_verification_queue.map(r => [r.id, r.item, r.location, r.placeholder, r.status]);
const author = addTable("Author Queue", "Author Verification Queue", "Unknown author-side metadata is never guessed and remains concentrated here.",
  ["ID", "Item", "Location", "Required placeholder", "Status"], authorData, [10, 36, 28, 70, 14], "E");
author.getRange(`D5:D${authorData.length + 4}`).format = { fill: AMBER, wrapText: true };

const gates = wb.worksheets.add("Gates");
title(gates, "Architecture completion gates", "A PASS here means the architecture is ready for drafting; sentence-level citation checks still repeat after writing.", "D");
const gateRows = [
  ["Frozen identity", "21,648 results; ppo_mlp excluded", "PASS", "F01-F04"],
  ["Primary endpoint", "safe_weighted_coverage retained", "PASS", "F02"],
  ["Claim coverage", `${source.claims.length} claims mapped`, "PASS", "Claim Map"],
  ["Figure story", `${source.figure_storyline.length} evidence units`, "PASS", "Figure Storyline"],
  ["Statistical unit", "map-level pairing; nested task/seed observations", "PASS", "K15"],
  ["Claim boundaries", "no flight, certification, equivalence or unseen-size claim", "PASS", "K04; K11; K13; K14"],
  ["Author unknowns", `${source.author_verification_queue.length} items centralised`, "OPEN BY DESIGN", "Author Queue"],
];
gates.getRange("A4:D11").values = [["Gate", "Criterion", "Status", "Evidence"], ...gateRows];
const gateTable = gates.tables.add("A4:D11", true, "ArchitectureGatesTable");
gates.freezePanes.freezeRows(4);
for (const [i, w] of [22, 70, 22, 30].entries()) gates.getRangeByIndexes(0, i, 11, 1).format.columnWidth = w;
gates.getRange("A5:D11").format = { rowHeight: 52, wrapText: true, verticalAlignment: "center" };
gates.getRange("C5:C10").format = { fill: GREEN, font: { bold: true } };
gates.getRange("C11").format = { fill: AMBER, font: { bold: true } };

const previewSheets = ["Fact Map", "Claim Map", "Evidence Matrix", "Figure Storyline", "Paragraph Plan", "Author Queue", "Gates"];
const reports = [];
for (const sheetName of previewSheets) {
  const preview = await wb.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  const previewPath = path.join(PREVIEW_DIR, `${sheetName.replaceAll(" ", "_")}.png`);
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  const inspection = await wb.inspect({ kind: "region", sheetId: sheetName, range: "A1:I100", maxChars: 5000 });
  reports.push({ sheetName, previewPath, inspection: inspection.ndjson || String(inspection) });
}
const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 200 } });
const outputFile = await SpreadsheetFile.exportXlsx(wb);
await outputFile.save(OUTPUT);
await fs.writeFile(QA_OUTPUT, JSON.stringify({ reports, errors: errors.ndjson || String(errors) }, null, 2), "utf8");
console.log(JSON.stringify({ output: OUTPUT, sheets: wb.worksheets.items.length, facts: source.facts.length, claims: source.claims.length }, null, 2));
