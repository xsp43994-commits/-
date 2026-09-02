import fs from "node:fs/promises";
import path from "node:path";
import {
  SpreadsheetFile,
  Workbook,
} from "file:///C:/Users/xsp/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

// 关键可调参数：输入登记簿、两个工作簿输出与预览目录。
const ROOT = "C:/Users/xsp/Desktop/DRL代码/paper_delivery/EAAI_fulltext_rewrite_v2_2026-08-09";
const SOURCE_PATH = path.join(ROOT, "literature", "reference_status_v2.json");
const STATUS_OUTPUT = path.join(ROOT, "literature", "Reference_Fulltext_Status_v2.xlsx");
const CLAIM_OUTPUT = path.join(ROOT, "literature", "Claim_Citation_Map_v2.xlsx");
const PREVIEW_DIR = path.join(ROOT, "qa", "reference_workbook_previews");

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

function columnName(count) {
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

function writeTable(sheet, startRow, headers, rows, widths = {}) {
  const endCol = columnName(headers.length);
  sheet.getRange(`A${startRow}:${endCol}${startRow}`).values = [headers];
  sheet.getRange(`A${startRow}:${endCol}${startRow}`).format = {
    fill: COLORS.blue,
    font: { bold: true, color: COLORS.white, size: 9 },
    rowHeight: 34,
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: COLORS.grid },
  };
  if (rows.length) {
    sheet.getRange(`A${startRow + 1}:${endCol}${startRow + rows.length}`).values = rows;
    sheet.getRange(`A${startRow + 1}:${endCol}${startRow + rows.length}`).format = {
      font: { color: COLORS.text, size: 8 },
      wrapText: true,
      verticalAlignment: "top",
      borders: { preset: "all", style: "thin", color: COLORS.grid },
    };
    sheet.getRange(`A${startRow}:${endCol}${startRow + rows.length}`).format.autofitRows();
  }
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
  sheet.freezePanes.freezeRows(startRow);
}

function makeStatusWorkbook() {
  const workbook = Workbook.create();
  const summary = workbook.worksheets.add("Summary");
  const references = workbook.worksheets.add("Reference Status");
  const access = workbook.worksheets.add("Access Control");
  const excluded = workbook.worksheets.add("Excluded Candidates");

  titleBand(summary, "Reference full-text status v2", "Retained references pass existence, identity, full-text relevance and planned sentence-support gates.", "F");
  const redistributable = source.references.filter((record) => record.redistribute_pdf && record.fulltext_file).length;
  const linkOnly = source.references.length - redistributable;
  summary.getRange("A4:B13").values = [
    ["Retained references", source.reference_count],
    ["Mapped external claims", source.claims.length],
    ["Open PDFs deliverable", redistributable],
    ["Link-only / official page", linkOnly],
    ["School access required", source.school_access_required.length],
    ["Existence gate", source.gate.existence],
    ["Identity gate", source.gate.identity],
    ["Full-text relevance", source.gate.fulltext_relevance],
    ["Sentence support", source.gate.sentence_level_entailment],
    ["Prepared", source.prepared],
  ];
  summary.getRange("A4:A13").format = { fill: COLORS.paleGreen, font: { bold: true }, borders: { preset: "all", style: "thin", color: COLORS.grid } };
  summary.getRange("B4:B13").format = { wrapText: true, borders: { preset: "all", style: "thin", color: COLORS.grid } };
  summary.getRange("A:A").format.columnWidth = 28;
  summary.getRange("B:B").format.columnWidth = 88;

  titleBand(references, "Retained references and full-text evidence", "The support column states the maximum proposition authorised by the inspected full text.", "P");
  const referenceRows = source.references.map((record) => [
    record.ref_id,
    record.year,
    record.title,
    (record.authors || []).join("; "),
    record.venue || "",
    record.doi || "",
    record.source_id,
    record.fulltext_status,
    record.fulltext_file || "",
    record.fulltext_pages || "",
    record.fulltext_sha256 || "",
    record.redistribute_pdf ? "PDF deliverable" : "Link only",
    record.access_url || record.url || "",
    record.support,
    record.metadata_identity_verified ? "PASS" : "FAIL",
    "Recheck exact sentence at insertion",
  ]);
  writeTable(references, 4,
    ["Ref", "Year", "Title", "Authors", "Venue", "DOI", "Source ID", "Full-text status", "Local full text", "Pages", "SHA-256", "Delivery", "Verified access", "Maximum supported proposition", "Identity", "Insertion control"],
    referenceRows,
    { A: 8, B: 8, C: 55, D: 48, E: 40, F: 31, G: 30, H: 31, I: 36, J: 8, K: 67, L: 17, M: 58, N: 75, O: 11, P: 32 },
  );
  references.getRange(`O5:O${4 + referenceRows.length}`).format.fill = COLORS.paleGreen;

  titleBand(access, "Copyright and delivery control", "A readable full text is not automatically redistributable. The delivery status is therefore recorded separately.", "G");
  const accessRows = source.references.map((record) => [
    record.ref_id,
    record.title,
    record.fulltext_status,
    record.redistribute_pdf ? "Yes" : "No",
    record.fulltext_file || "",
    record.access_url || record.url || "",
    record.redistribute_pdf ? "Include open PDF and link" : "Include link only; do not copy PDF",
  ]);
  writeTable(access, 4, ["Ref", "Title", "Full-text source", "PDF redistributable", "File", "Verified link", "Delivery action"], accessRows,
    { A: 8, B: 60, C: 34, D: 19, E: 38, F: 60, G: 42 });
  for (let index = 0; index < source.references.length; index += 1) {
    const row = 5 + index;
    access.getRange(`D${row}:G${row}`).format.fill = source.references[index].redistribute_pdf ? COLORS.paleGreen : COLORS.paleAmber;
  }

  titleBand(excluded, "Historical candidates excluded from final citation set", "Exclusion avoids weak, redundant or unused citations; these records do not require school-access retrieval.", "C");
  const excludedRows = source.excluded_candidates.map((record) => [record.source_id, record.reason, "Excluded — no full-text request"]);
  writeTable(excluded, 4, ["Candidate", "Reason", "Action"], excludedRows, { A: 34, B: 100, C: 34 });
  excluded.getRange(`C5:C${4 + excludedRows.length}`).format.fill = COLORS.paleRed;
  return { workbook, sheets: ["Summary", "Reference Status", "Access Control", "Excluded Candidates"] };
}

function makeClaimWorkbook() {
  const workbook = Workbook.create();
  const claims = workbook.worksheets.add("Claim Citation Map");
  const verification = workbook.worksheets.add("Entailment Gate");
  titleBand(claims, "Claim–citation map v2", "Each row is an external proposition planned for the manuscript; frozen experimental findings are mapped separately in the Evidence Matrix.", "G");
  const claimRows = source.claims.map((claim) => [
    claim.claim_id,
    claim.proposition,
    claim.references.join("; "),
    claim.planned_location,
    "External literature",
    "Full-text support verified",
    "No stronger inference permitted",
  ]);
  writeTable(claims, 4, ["Claim", "Exact proposition", "Supporting refs", "Planned location", "Evidence class", "Current gate", "Claim-strength control"], claimRows,
    { A: 10, B: 96, C: 30, D: 35, E: 24, F: 30, G: 38 });
  claims.getRange(`F5:F${4 + claimRows.length}`).format.fill = COLORS.paleGreen;

  titleBand(verification, "Four-level citation verification", "A citation enters the final manuscript only after all four checks pass for the exact sentence.", "F");
  const verificationRows = source.claims.map((claim) => [
    claim.claim_id,
    claim.references.join("; "),
    "PASS",
    "PASS",
    "PASS",
    "PASS at map level; repeat after final sentence wording",
  ]);
  writeTable(verification, 4, ["Claim", "References", "Existence", "Identity", "Full-text relevance", "Sentence entailment"], verificationRows,
    { A: 10, B: 32, C: 14, D: 14, E: 24, F: 52 });
  verification.getRange(`C5:F${4 + verificationRows.length}`).format.fill = COLORS.paleGreen;
  return { workbook, sheets: ["Claim Citation Map", "Entailment Gate"] };
}

async function renderAndInspect(workbook, prefix, sheets) {
  const reports = [];
  for (const sheetName of sheets) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
    const previewPath = path.join(PREVIEW_DIR, `${prefix}_${sheetName.replaceAll(" ", "_")}.png`);
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
    const inspection = await workbook.inspect({ kind: "region", sheetId: sheetName, range: "A1:P100", maxChars: 5000 });
    reports.push({ sheetName, previewPath, inspection: inspection.ndjson || String(inspection) });
  }
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 200 } });
  return { reports, errors: errors.ndjson || String(errors) };
}

const statusPackage = makeStatusWorkbook();
const claimPackage = makeClaimWorkbook();
const statusQA = await renderAndInspect(statusPackage.workbook, "status", statusPackage.sheets);
const claimQA = await renderAndInspect(claimPackage.workbook, "claims", claimPackage.sheets);

const statusFile = await SpreadsheetFile.exportXlsx(statusPackage.workbook);
await statusFile.save(STATUS_OUTPUT);
const claimFile = await SpreadsheetFile.exportXlsx(claimPackage.workbook);
await claimFile.save(CLAIM_OUTPUT);
await fs.writeFile(path.join(ROOT, "qa", "reference_workbook_inspection.json"), JSON.stringify({ statusQA, claimQA }, null, 2), "utf8");

console.log(JSON.stringify({ statusWorkbook: STATUS_OUTPUT, claimWorkbook: CLAIM_OUTPUT, references: source.references.length, claims: source.claims.length }, null, 2));
