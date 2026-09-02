import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "file:///C:/Users/xsp/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";


// 关键路径参数：只读取文献核验 JSON，输出两个可编辑工作簿与逐表预览图。
const DELIVERY_ROOT = "C:/Users/xsp/Desktop/DRL代码/paper_delivery/EAAI_2026-08-09";
const REGISTER_PATH = path.join(DELIVERY_ROOT, "literature", "literature_register.json");
const PREVIEW_DIR = path.join(DELIVERY_ROOT, "qa", "workbook_previews");
const register = JSON.parse(await fs.readFile(REGISTER_PATH, "utf8"));
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
};

function lastColumnName(count) {
  let n = count;
  let name = "";
  while (n > 0) {
    n -= 1;
    name = String.fromCharCode(65 + (n % 26)) + name;
    n = Math.floor(n / 26);
  }
  return name;
}

function writeTable(sheet, startRow, headers, rows, widths = {}) {
  const endCol = lastColumnName(headers.length);
  sheet.getRange(`A${startRow}:${endCol}${startRow}`).values = [headers];
  sheet.getRange(`A${startRow}:${endCol}${startRow}`).format = {
    fill: COLORS.navy,
    font: { bold: true, color: "#FFFFFF", size: 10 },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: COLORS.grid },
  };
  if (rows.length) {
    sheet.getRange(`A${startRow + 1}:${endCol}${startRow + rows.length}`).values = rows;
    sheet.getRange(`A${startRow + 1}:${endCol}${startRow + rows.length}`).format = {
      font: { color: COLORS.text, size: 9 },
      wrapText: true,
      verticalAlignment: "top",
      borders: { preset: "all", style: "thin", color: COLORS.grid },
    };
  }
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
  sheet.freezePanes.freezeRows(startRow);
  sheet.showGridLines = false;
}

function titleBand(sheet, title, subtitle, endCol = "H") {
  sheet.getRange(`A1:${endCol}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${endCol}1`).format = {
    fill: COLORS.navy,
    font: { bold: true, color: "#FFFFFF", size: 16 },
    rowHeight: 28,
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${endCol}2`).merge();
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${endCol}2`).format = {
    fill: COLORS.paleBlue,
    font: { italic: true, color: COLORS.text, size: 9 },
    wrapText: true,
    rowHeight: 34,
    verticalAlignment: "center",
  };
}

async function renderWorkbook(workbook, prefix, sheetNames) {
  const reports = [];
  for (const sheetName of sheetNames) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
    const file = path.join(PREVIEW_DIR, `${prefix}_${sheetName.replaceAll(" ", "_")}.png`);
    await fs.writeFile(file, new Uint8Array(await preview.arrayBuffer()));
    const inspection = await workbook.inspect({ kind: "region", sheetId: sheetName, range: "A1:Z80", maxChars: 1600 });
    reports.push({ sheetName, preview: file, inspection: inspection.ndjson || String(inspection) });
  }
  return reports;
}

function createExemplarsWorkbook() {
  const workbook = Workbook.create();
  const summary = workbook.worksheets.add("Summary");
  const exemplars = workbook.worksheets.add("Exemplars");
  const functions = workbook.worksheets.add("Style Functions");

  titleBand(summary, "EAAI Exemplar Register", "Twelve 2022–2026 Original Research exemplars. Structural functions are extracted without copying language, data, or reference combinations.", "F");
  summary.getRange("A4:B10").values = [
    ["Target journal", "Engineering Applications of Artificial Intelligence"],
    ["Article type", "Original Research"],
    ["Exemplar count", register.exemplars.length],
    ["Metadata gate", "DOI + Crossref metadata verified"],
    ["Content gate", "Publisher abstract/page or legal full text; limitations recorded"],
    ["Writing rule", "Use section function and evidence order only; never copy wording"],
    ["Prepared", register.register_created],
  ];
  summary.getRange("A4:A10").format = { fill: COLORS.paleBlue, font: { bold: true }, borders: { preset: "all", style: "thin", color: COLORS.grid } };
  summary.getRange("B4:B10").format = { wrapText: true, borders: { preset: "all", style: "thin", color: COLORS.grid } };
  summary.getRange("A:A").format.columnWidth = 22;
  summary.getRange("B:B").format.columnWidth = 75;
  summary.showGridLines = false;

  titleBand(exemplars, "Twelve EAAI Original Research Exemplars", "Legal access is recorded conservatively; a DOI link is supplied whenever a PDF cannot be redistributed.", "L");
  const exemplarHeaders = ["No.", "Year", "Title", "Authors", "DOI", "Volume", "Article / pages", "Publisher link", "Scientific similarity", "Style use", "Verification level", "Legal access"];
  const exemplarRows = register.exemplars.map((x, i) => [
    i + 1, x.year, x.title, x.authors.join("; "), x.doi, x.volume || "", x.article_number_or_pages || "",
    `https://doi.org/${x.doi}`, x.scientific_function, "Argument order, section function, result density, and figure narrative only", x.verification_level, x.legal_access,
  ]);
  writeTable(exemplars, 4, exemplarHeaders, exemplarRows, { A: 6, B: 9, C: 52, D: 38, E: 30, F: 10, G: 16, H: 35, I: 48, J: 42, K: 48, L: 32 });
  exemplars.getRange(`B5:B${4 + exemplarRows.length}`).format.numberFormat = "0";

  titleBand(functions, "Reusable Scientific Functions", "A journal-style profile derived from the exemplars and the official author guide. No exemplar sentence is reproduced.", "F");
  const functionRows = [
    ["Abstract", "State engineering problem, AI novelty, evaluation scope, quantitative outcomes, and bounded conclusion", "≤250 words; define AI and engineering contributions separately", "Main manuscript abstract"],
    ["Introduction", "Move from engineering constraints to algorithmic gap and testable contributions", "Avoid generic UAV hype; end with three evidence-bounded contributions", "Sections 1–2"],
    ["Problem formulation", "Define fixed points, priority, resource budgets, depot return, and legal action set", "Keep engineering quantities and symbols traceable", "Section 3"],
    ["Method", "Explain encoder, return-aware mask, pointer decoder, critic, and PPO training", "Separate feasibility mechanism from reward shaping", "Section 4"],
    ["Experiments", "Describe frozen split, baselines, map-level statistics, robustness, transfer, and ablations", "Map is the independent unit; identify zero-shot simulation transfer", "Section 5"],
    ["Results", "Present coverage, safety, online time, training evidence, transfer, robustness, and ablation", "Report ACO/SA/MILP strengths instead of declaring a universal winner", "Section 6"],
    ["Discussion", "Distinguish observation, interpretation, speculation, applicability, and limitations", "No real-flight or safety-certification claims", "Section 7"],
    ["Data statement", "Specify frozen package, source data, reconstruction, licensing, and anonymous access", "Use author placeholder until permanent anonymous repository exists", "End matter"],
  ];
  writeTable(functions, 4, ["Section function", "Expected reasoning role", "EAAI-specific control", "Use in this paper"], functionRows, { A: 20, B: 58, C: 55, D: 28 });
  return { workbook, sheetNames: ["Summary", "Exemplars", "Style Functions"] };
}

function createReferencesWorkbook() {
  const workbook = Workbook.create();
  const summary = workbook.worksheets.add("Summary");
  const refs = workbook.worksheets.add("References");
  const claims = workbook.worksheets.add("Claim Citation Map");
  const verification = workbook.worksheets.add("Verification Log");

  titleBand(summary, "Verified Reference Register", "Reference count follows actual manuscript claims. Metadata verification is distinct from sentence-level support.", "F");
  summary.getRange("A4:B11").values = [
    ["Reference records", register.references.length],
    ["Claim groups", register.claim_citation_map.length],
    ["Open PDFs downloaded", register.open_access_download_log.filter((x) => x.status === "downloaded").length],
    ["RIS file", "verified_references.ris"],
    ["DOI gate", "Resolved metadata or an official/manual record"],
    ["Claim gate", "Use only after abstract/full-text inspection supports the exact proposition"],
    ["Copyright gate", "Redistribute only legally open PDFs; otherwise give DOI/publisher links"],
    ["Prepared", register.register_created],
  ];
  summary.getRange("A4:A11").format = { fill: COLORS.paleGreen, font: { bold: true }, borders: { preset: "all", style: "thin", color: COLORS.grid } };
  summary.getRange("B4:B11").format = { wrapText: true, borders: { preset: "all", style: "thin", color: COLORS.grid } };
  summary.getRange("A:A").format.columnWidth = 24;
  summary.getRange("B:B").format.columnWidth = 82;
  summary.showGridLines = false;

  titleBand(refs, "Reference Records", "DOI-normalised and deduplicated. Manual open records retain official URLs when no Crossref DOI record is available.", "M");
  const refHeaders = ["No.", "ID", "Year", "Title", "Authors", "Venue", "Volume", "Issue", "Article / pages", "DOI", "URL", "Verification", "Access / licence"];
  const refRows = register.references.map((x, i) => [
    i + 1, x.doi || x.id, x.year, x.title, (x.authors || []).join("; "), x.venue || "", x.volume || "", x.issue || "",
    x.article_number_or_pages || "", x.doi || "", x.url || "", x.verification || "", [...(x.licences || []), ...(x.publisher_pdf_links || [])].join("; ") || (x.pdf_url ? "Open PDF listed" : "Link only"),
  ]);
  writeTable(refs, 4, refHeaders, refRows, { A: 6, B: 30, C: 8, D: 55, E: 42, F: 34, G: 9, H: 8, I: 16, J: 30, K: 38, L: 48, M: 38 });

  titleBand(claims, "Claim–Citation Map", "Each external proposition has an explicit supporting-reference set. Frozen experimental findings are mapped separately in the Evidence Matrix.", "E");
  const claimRows = register.claim_citation_map.map((x) => [x.claim_id, x.claim, x.reference_ids.join("; "), "External literature", "Checked before manuscript insertion"]);
  writeTable(claims, 4, ["Claim ID", "Exact proposition", "Supporting reference IDs", "Evidence class", "Use control"], claimRows, { A: 12, B: 78, C: 64, D: 22, E: 35 });

  titleBand(verification, "Verification and Access Log", "A downloaded file is never assumed to be redistributable unless it comes from a clearly legal open source.", "F");
  const verRows = register.open_access_download_log.map((x) => [x.id, x.status, x.url, x.file || "", x.bytes || 0, x.error || ""]);
  writeTable(verification, 4, ["Reference ID", "Status", "Source URL", "Local file", "Bytes", "Error / note"], verRows, { A: 28, B: 16, C: 62, D: 62, E: 14, F: 45 });
  verification.getRange(`E5:E${4 + Math.max(1, verRows.length)}`).format.numberFormat = "#,##0";
  return { workbook, sheetNames: ["Summary", "References", "Claim Citation Map", "Verification Log"] };
}

const exemplarPackage = createExemplarsWorkbook();
const referencePackage = createReferencesWorkbook();

const reports = [];
reports.push(...await renderWorkbook(exemplarPackage.workbook, "exemplars", exemplarPackage.sheetNames));
reports.push(...await renderWorkbook(referencePackage.workbook, "references", referencePackage.sheetNames));

const exemplarFile = await SpreadsheetFile.exportXlsx(exemplarPackage.workbook);
await exemplarFile.save(path.join(DELIVERY_ROOT, "literature", "EAAI_exemplars_register.xlsx"));
const referenceFile = await SpreadsheetFile.exportXlsx(referencePackage.workbook);
await referenceFile.save(path.join(DELIVERY_ROOT, "literature", "verified_references.xlsx"));

await fs.writeFile(path.join(DELIVERY_ROOT, "qa", "workbook_inspection.json"), JSON.stringify(reports, null, 2), "utf8");
console.log(JSON.stringify({ workbooks: 2, renderedSheets: reports.length }, null, 2));
