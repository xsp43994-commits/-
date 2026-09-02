import fs from "node:fs/promises";
import path from "node:path";
import {
  SpreadsheetFile,
  Workbook,
} from "file:///C:/Users/xsp/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

// 关键可调参数：第二轮根目录、源数据文件和预览缩放统一放在这里，避免散落修改。
const DELIVERY_ROOT = "C:/Users/xsp/Desktop/DRL代码/paper_delivery/EAAI_fulltext_rewrite_v2_2026-08-09";
const SOURCE_PATH = path.join(DELIVERY_ROOT, "literature", "eaai_12_fulltext_matrix_source.json");
const OUTPUT_PATH = path.join(DELIVERY_ROOT, "literature", "EAAI_12_Fulltext_Reading_Matrix.xlsx");
const PREVIEW_DIR = path.join(DELIVERY_ROOT, "qa", "fulltext_matrix_previews");
const PREVIEW_SCALE = 1;

const source = JSON.parse(await fs.readFile(SOURCE_PATH, "utf8"));
await fs.mkdir(PREVIEW_DIR, { recursive: true });

const COLORS = {
  navy: "#17365D",
  blue: "#2F75B5",
  paleBlue: "#D9EAF7",
  paleGreen: "#E2F0D9",
  paleAmber: "#FFF2CC",
  paleRed: "#FCE4D6",
  grid: "#D9E2F3",
  text: "#1F2937",
  white: "#FFFFFF",
};

function colName(count) {
  let n = count;
  let result = "";
  while (n > 0) {
    n -= 1;
    result = String.fromCharCode(65 + (n % 26)) + result;
    n = Math.floor(n / 26);
  }
  return result;
}

function titleBand(sheet, title, subtitle, endCol) {
  sheet.getRange(`A1:${endCol}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${endCol}1`).format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 15 },
    rowHeight: 28,
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${endCol}2`).merge();
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${endCol}2`).format = {
    fill: COLORS.paleBlue,
    font: { italic: true, color: COLORS.text, size: 9 },
    rowHeight: 40,
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.showGridLines = false;
}

function table(sheet, startRow, headers, rows, widths = {}) {
  const endCol = colName(headers.length);
  sheet.getRange(`A${startRow}:${endCol}${startRow}`).values = [headers];
  sheet.getRange(`A${startRow}:${endCol}${startRow}`).format = {
    fill: COLORS.blue,
    font: { bold: true, color: COLORS.white, size: 9 },
    wrapText: true,
    verticalAlignment: "center",
    rowHeight: 34,
    borders: { preset: "all", style: "thin", color: COLORS.grid },
  };
  if (rows.length > 0) {
    sheet.getRange(`A${startRow + 1}:${endCol}${startRow + rows.length}`).values = rows;
    sheet.getRange(`A${startRow + 1}:${endCol}${startRow + rows.length}`).format = {
      font: { color: COLORS.text, size: 8 },
      wrapText: true,
      verticalAlignment: "top",
      borders: { preset: "all", style: "thin", color: COLORS.grid },
    };
  }
  Object.entries(widths).forEach(([column, width]) => {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  });
  sheet.freezePanes.freezeRows(startRow);
  sheet.getRange(`A${startRow}:${endCol}${startRow + rows.length}`).format.autofitRows();
}

const workbook = Workbook.create();

const summary = workbook.worksheets.add("Summary");
titleBand(
  summary,
  "EAAI Full-text Reading Matrix v2",
  "All twelve acquired PDFs were read in full. This workbook records scientific functions and non-copy boundaries; it is not a sentence bank.",
  "F",
);
summary.getRange("A4:B13").values = [
  ["Target journal", "Engineering Applications of Artificial Intelligence"],
  ["Article type", "Research article / Original Research"],
  ["Full texts acquired", source.papers.length],
  ["Full-text records complete", source.papers.filter((paper) => paper.status === "Complete").length],
  ["Total PDF pages", source.papers.reduce((sum, paper) => sum + paper.pages, 0)],
  ["S-High patterns", source.style_patterns.filter((pattern) => pattern.strength === "S-High").length],
  ["S-Medium patterns", source.style_patterns.filter((pattern) => pattern.strength === "S-Medium").length],
  ["O/Low patterns", source.style_patterns.filter((pattern) => pattern.strength === "O/Low").length],
  ["Strength gate", source.strength_rule],
  ["Prepared", source.prepared],
];
summary.getRange("A4:A13").format = {
  fill: COLORS.paleGreen,
  font: { bold: true, color: COLORS.text },
  borders: { preset: "all", style: "thin", color: COLORS.grid },
};
summary.getRange("B4:B13").format = {
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "all", style: "thin", color: COLORS.grid },
};
summary.getRange("A:A").format.columnWidth = 26;
summary.getRange("B:B").format.columnWidth = 92;
summary.getRange("A15:F15").merge();
summary.getRange("A15").values = [["Interpretation control"]];
summary.getRange("A15:F15").format = { fill: COLORS.paleAmber, font: { bold: true }, rowHeight: 22 };
summary.getRange("A16:F18").merge();
summary.getRange("A16").values = [[
  "A frequent pattern is not an official author-guide requirement. It becomes a working style convention only after passing the frequency/comparability gate. Paper-specific language, equations, data, parameters, figure layouts and citation clusters remain prohibited.",
]];
summary.getRange("A16:F18").format = { wrapText: true, verticalAlignment: "top", borders: { preset: "all", style: "thin", color: COLORS.grid } };

const reading = workbook.worksheets.add("Reading Matrix");
titleBand(reading, "Twelve complete-paper reading records", "Each row summarises one complete-PDF reading. Use the Markdown note for finer paper-level detail.", "Q");
const readingHeaders = [
  "No.", "Year", "Title", "DOI", "Pages", "SHA-256", "Article type", "Comparability",
  "Introduction move", "Methods granularity", "Results logic", "Discussion / claim boundary",
  "Figure and table role", "Citation function", "Uncertainty and claim strength", "Non-copy boundary", "Status",
];
const readingRows = source.papers.map((paper) => [
  paper.no, paper.year, paper.title, paper.doi, paper.pages, paper.sha256, paper.article_type, paper.comparability,
  paper.introduction_move, paper.methods_granularity, paper.results_logic, paper.discussion_boundary,
  paper.figure_role, paper.citation_function, paper.uncertainty_claim_strength, paper.noncopy, paper.status,
]);
table(reading, 4, readingHeaders, readingRows, {
  A: 6, B: 8, C: 52, D: 30, E: 8, F: 67, G: 16, H: 16, I: 46, J: 52, K: 50,
  L: 52, M: 48, N: 46, O: 50, P: 54, Q: 12,
});
reading.getRange(`A5:B${4 + readingRows.length}`).format.numberFormat = "0";
reading.getRange(`E5:E${4 + readingRows.length}`).format.numberFormat = "0";
reading.getRange(`Q5:Q${4 + readingRows.length}`).format.fill = COLORS.paleGreen;

const patterns = workbook.worksheets.add("Style Patterns");
titleBand(patterns, "Cross-paper style-frequency audit", source.strength_rule, "I");
const patternHeaders = ["ID", "Pattern", "Supporting papers", "Count", "Comparable N", "Rate", "Strength", "Manuscript use", "Gate check"];
const patternRows = source.style_patterns.map((pattern, index) => {
  return [
    pattern.id,
    pattern.pattern,
    pattern.papers,
    pattern.count,
    pattern.comparable_n,
    pattern.rate,
    pattern.strength,
    pattern.use,
    "",
  ];
});
table(patterns, 4, patternHeaders, patternRows, { A: 10, B: 68, C: 27, D: 10, E: 15, F: 12, G: 14, H: 52, I: 24 });
patterns.getRange(`D5:E${4 + patternRows.length}`).format.numberFormat = "0";
patterns.getRange(`F5:F${4 + patternRows.length}`).format.numberFormat = "0.0%";
// 公式单独写入公式属性，避免被工作簿引擎当作普通对象文本。
patterns.getRange(`I5:I${4 + patternRows.length}`).formulas = source.style_patterns.map((_, index) => {
  const row = 5 + index;
  return [`=IF(AND(D${row}>=5,F${row}>=0.7),"Pass S-High gate","Below S-High gate")`];
});
for (let index = 0; index < source.style_patterns.length; index += 1) {
  const row = 5 + index;
  const strength = source.style_patterns[index].strength;
  const fill = strength === "S-High" ? COLORS.paleGreen : strength === "S-Medium" ? COLORS.paleAmber : COLORS.paleRed;
  patterns.getRange(`G${row}:I${row}`).format.fill = fill;
}

const noncopy = workbook.worksheets.add("Noncopy Boundaries");
titleBand(noncopy, "Paper-specific content that must not migrate", "Only recurring scientific functions may influence the v2 manuscript.", "D");
const noncopyRows = source.papers.map((paper) => [paper.no, paper.doi, paper.title, paper.noncopy]);
table(noncopy, 4, ["No.", "DOI", "Paper", "Prohibited migration"], noncopyRows, { A: 6, B: 30, C: 58, D: 86 });

const verification = workbook.worksheets.add("Verification Log");
titleBand(verification, "Full-text completion and identity audit", "Page counts and hashes identify the exact acquired PDFs used for the reading records.", "H");
const verificationRows = source.papers.map((paper) => [
  paper.no,
  paper.doi,
  paper.pages,
  paper.sha256,
  paper.article_type,
  paper.comparability,
  paper.status,
  "Full PDF, including figures/tables, declarations and references, reviewed",
]);
table(verification, 4, ["No.", "DOI", "Pages", "SHA-256", "Article type", "Comparability", "Status", "Coverage statement"], verificationRows, {
  A: 6, B: 30, C: 9, D: 67, E: 18, F: 18, G: 12, H: 70,
});
verification.getRange(`G5:G${4 + verificationRows.length}`).format.fill = COLORS.paleGreen;

const sheetNames = ["Summary", "Reading Matrix", "Style Patterns", "Noncopy Boundaries", "Verification Log"];
const inspections = [];
for (const sheetName of sheetNames) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: PREVIEW_SCALE, format: "png" });
  const previewPath = path.join(PREVIEW_DIR, `${sheetName.replaceAll(" ", "_")}.png`);
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  const inspection = await workbook.inspect({ kind: "region", sheetId: sheetName, range: "A1:Q80", maxChars: 5000 });
  inspections.push({ sheetName, previewPath, inspection: inspection.ndjson || String(inspection) });
}

const formulaInspection = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 200 },
  summary: "final formula error scan",
});

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(OUTPUT_PATH);
await fs.writeFile(
  path.join(DELIVERY_ROOT, "qa", "fulltext_matrix_inspection.json"),
  JSON.stringify({ inspections, formulaInspection: formulaInspection.ndjson || String(formulaInspection) }, null, 2),
  "utf8",
);

console.log(JSON.stringify({ output: OUTPUT_PATH, sheets: sheetNames.length, papers: source.papers.length }, null, 2));
