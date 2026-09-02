import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outDir = path.join(root, "workbooks");
const qaDir = path.join(root, "qa", "workbook_previews");
await fs.mkdir(outDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });

const navy = "#17365D", pale = "#D9EAF7", green = "#E2F0D9", amber = "#FFF2CC", ink = "#1F2937", grid = "#D9E2F3";

function title(sheet, text, subtitle, lastCol) {
  sheet.mergeCells(`A1:${lastCol}1`); sheet.getRange("A1").values = [[text]];
  sheet.getRange(`A1:${lastCol}1`).format = { fill: navy, font: { bold: true, color: "#FFFFFF", size: 15 }, rowHeight: 27 };
  sheet.mergeCells(`A2:${lastCol}2`); sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${lastCol}2`).format = { fill: pale, font: { italic: true, color: ink, size: 9 }, wrapText: true, rowHeight: 34 };
  sheet.showGridLines = false;
}

function header(range) {
  range.format = { fill: navy, font: { bold: true, color: "#FFFFFF" }, wrapText: true, verticalAlignment: "center", borders: { preset: "all", style: "thin", color: grid } };
}

function body(range) {
  range.format = { font: { color: ink, size: 9 }, wrapText: true, verticalAlignment: "top", borders: { preset: "all", style: "thin", color: grid } };
}

async function exportAndRender(wb, filename, sheets) {
  const blob = await SpreadsheetFile.exportXlsx(wb); await blob.save(path.join(outDir, filename));
  for (const [name, range] of sheets) {
    const preview = await wb.render({ sheetName: name, range, autoCrop: "all", scale: 1.35, format: "png" });
    await fs.writeFile(path.join(qaDir, `${filename.replace(".xlsx", "")}_${name.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
  const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula errors" });
  await fs.writeFile(path.join(qaDir, `${filename}.errors.ndjson`), errors.ndjson ?? "", "utf8");
}

const papers = [
  [1,2026,"Deep reinforcement learning for periodic UAV task allocation and path planning with a fixed nest station","10.1016/j.engappai.2026.115219",16,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Integrated results and discussion","Declarations → references","O: published two-column only"],
  [2,2026,"Asynchronous multithreading reinforcement learning with attention-based significance measurement for collision-free robot navigation","10.1016/j.engappai.2026.113779",17,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Separate discussion/validation boundary","Declarations → references","O: publisher front matter omitted"],
  [3,2025,"Situation-aware deep reinforcement learning for UAV swarm navigation in dynamic multi-obstacle environments","10.1016/j.engappai.2025.113518",20,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Results then synthesis/discussion","Declarations → references","S: evidence-rich result sequence"],
  [4,2025,"Evaluating reinforcement learning-based neural controllers for quadcopter navigation in windy conditions","10.1016/j.engappai.2025.112090",18,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Results → discussion/limitations","Declarations → references","S: negative findings retained"],
  [5,2025,"Human-in-the-loop reinforcement learning for dynamic soaring","10.1016/j.engappai.2025.111219",19,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Dedicated limitations","Declarations → references","S: deployment boundary"],
  [6,2025,"RL-based multi-perspective motion planning of manned eVTOL in urban wind fields","10.1016/j.engappai.2025.110392",17,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Results → limitations","Declarations → references","S: comparator wins visible"],
  [7,2024,"Path planning via reinforcement learning with closed-loop motion control and field tests","10.1016/j.engappai.2024.109870",13,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Simulation and field claims separated","Declarations → references","S: validation tiers"],
  [8,2024,"Layered learning in a quadrotor drone using FOPID and PPO","10.1016/j.engappai.2024.108926",19,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Results with limitations","Declarations → references","O: paper-specific layered layout"],
  [9,2024,"RL robot navigation using illegal actions for autonomous docking","10.1016/j.engappai.2024.108506",20,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Tiered results and discussion","Declarations → references","S: action-mask evidence"],
  [10,2024,"Reliable traversability learning based on demonstrated risk-cost mapping","10.1016/j.engappai.2024.109339",10,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Results → limitations","Declarations → references","S: engineering trade-off"],
  [11,2023,"Subtask-masked curriculum learning for UAV maneuver decision-making","10.1016/j.engappai.2023.106703",14,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Question-led results → limitations","Declarations → references","S: uncertainty and transfer limits"],
  [12,2023,"Multi-UAV trajectory optimizer for wireless data harvesting","10.1016/j.engappai.2023.105891",11,"Yes","Title → abstract → keywords","Numbered, 1–3 levels","When applicable","Below figure","Above table","Results → trade-offs","Declarations → references","S: resource-aware narrative"],
];

const rules = [
  ["H01","H","Official guide","Single-column Word author manuscript","Apply to EAAI_manuscript_anonymized_v3.docx","Applied"],
  ["H02","H","Official guide","Non-structured abstract; ≤250 words; 1–6 keywords","Keep abstract and keywords adjacent","Applied"],
  ["H03","H","Double-anonymized review","Remove authors, affiliations and identity-bearing headers from manuscript","Title page remains separate","Applied"],
  ["S01","S","12/12 comparable","Numbered section hierarchy with sentence-case headings","Retain eight-section scientific structure","Applied"],
  ["S02","S","12/12","Figures near claims; captions below","Eight evidence units placed at first substantive discussion","Applied"],
  ["S03","S","12/12","Table titles above; compact horizontal rules and repeated headers","Use author-manuscript tables, not publisher artwork","Applied"],
  ["S04","S","12/12","Equations editable and numbered where referred to","SWC inserted as editable Word equation (1)","Applied"],
  ["S05","S","11/12","Dedicated limitations and bounded deployment claims","Keep Discussion and limitations section","Applied"],
  ["O01","O","12/12 published PDFs","Two-column publisher typesetting","Use only in separately labelled reading proof","Preview only"],
  ["O02","O","Publisher production","Elsevier logo, article information, DOI header, volume/issue","Do not reproduce in author manuscript or proof","Excluded"],
  ["O03","O","Paper-specific","Exact panel sizing, running heads and reference line breaks","Do not imitate; let author manuscript reflow","Excluded"],
];

const applications = [
  ["Page size","A4","H / plan v3","English and Chinese single-column manuscripts"],
  ["Body text","Times New Roman 11 pt, 1.5 spacing","Plan v3","English submission manuscript"],
  ["Chinese body","宋体 + Times New Roman for Latin/numerals","Plan v3","Chinese author-review manuscript"],
  ["Headings","Black, left aligned, sentence case","S + plan v3","All v3 documents"],
  ["Figures","Near first substantive citation; caption below","S02","Main and supplement"],
  ["Tables","Title above; repeated header; restrained rules","S03","Main and supplement"],
  ["Equation","Editable OMML, centered, number at right","S04","SWC equation (1)"],
  ["References","Author–year, hanging indent, original English metadata","H / translation rule","English and Chinese manuscripts"],
  ["Reading proof","English two-column; explicit non-publisher header","O01 boundary","Not for submission"],
  ["Prohibited production marks","No logo, article info, fake DOI or volume/issue","O02","All author-produced files"],
];

const formatWb = Workbook.create();
// 所有跨表公式引用的工作表必须先创建，再写入公式。
const sum = formatWb.worksheets.add("Summary");
const pf = formatWb.worksheets.add("Paper Format");
const rs = formatWb.worksheets.add("Rules");
const ap = formatWb.worksheets.add("Application");
title(sum,"EAAI 12-paper format comparison v3","H = official hard rule; S = stable recurring convention; O = publisher or paper-specific feature. Scientific content and sentences are never copied.","F");
sum.getRange("A4:B9").values = [["Metric","Value"],["Papers reviewed",null],["Total PDF pages",null],["Published two-column PDFs",null],["H rules",null],["S rules",null]];
sum.getRange("B5").formulas = [["=COUNTA('Paper Format'!A5:A16)"]]; sum.getRange("B6").formulas = [["=SUM('Paper Format'!E5:E16)"]]; sum.getRange("B7").formulas = [["=COUNTIF('Paper Format'!F5:F16,\"Yes\")"]]; sum.getRange("B8").formulas = [["=COUNTIF('Rules'!B5:B15,\"H\")"]]; sum.getRange("B9").formulas = [["=COUNTIF('Rules'!B5:B15,\"S\")"]];
header(sum.getRange("A4:B4")); body(sum.getRange("A5:B9")); sum.getRange("A5:A9").format.fill = green; sum.getRange("A5:A9").format.font = {bold:true,color:ink}; sum.getRange("A:B").format.columnWidth = 28; sum.getRange("B:B").format.columnWidth = 55;
sum.mergeCells("A11:F11"); sum.getRange("A11").values = [["Decision boundary"]]; sum.getRange("A11:F11").format = {fill:amber,font:{bold:true,color:ink}};
sum.mergeCells("A12:F14"); sum.getRange("A12").values = [["The 12 papers are publisher-formatted two-column articles, but the EAAI author guide controls submission. Therefore the v3 single-column manuscript is the submission file; the two-column file is an explicitly labelled reading proof. Publisher branding and article metadata are excluded."]]; sum.getRange("A12:F14").format = {wrapText:true,verticalAlignment:"top",font:{size:10,color:ink}};

title(pf,"Paper-level format observations","Published PDF layout is recorded as evidence but not treated as a submission template.","N");
const pfHead=["No.","Year","Short title","DOI","Pages","Published 2-col","Front order","Heading hierarchy","Equation numbering","Figure caption","Table title","Results / discussion","Back matter","Transfer class"];
pf.getRange("A4:N4").values=[pfHead]; pf.getRange("A5:N16").values=papers; header(pf.getRange("A4:N4")); body(pf.getRange("A5:N16")); pf.freezePanes.freezeRows(4); pf.getRange("A:N").format.columnWidth=15; pf.getRange("C:C").format.columnWidth=45; pf.getRange("D:D").format.columnWidth=33; for(const c of ["G","H","I","J","K","L","M","N"]) pf.getRange(`${c}:${c}`).format.columnWidth=22; pf.getRange("A5:B16").format.horizontalAlignment="center"; pf.getRange("E5:F16").format.horizontalAlignment="center";

title(rs,"Format rules and transfer gate","Only H and sufficiently stable S items control the author manuscript. O items are excluded or confined to the reading proof.","F");
rs.getRange("A4:F4").values=[["ID","Class","Evidence","Observed / required format","v3 application","Status"]]; rs.getRange("A5:F15").values=rules; header(rs.getRange("A4:F4")); body(rs.getRange("A5:F15")); rs.freezePanes.freezeRows(4); rs.getRange("A:F").format.columnWidth=20; rs.getRange("C:C").format.columnWidth=25; rs.getRange("D:E").format.columnWidth=45; rs.getRange("B5:B15").conditionalFormats.add("containsText",{text:"H",format:{fill:"#FCE4D6",font:{bold:true,color:"#9C0006"}}}); rs.getRange("B5:B15").conditionalFormats.add("containsText",{text:"S",format:{fill:green,font:{bold:true,color:"#006100"}}}); rs.getRange("B5:B15").conditionalFormats.add("containsText",{text:"O",format:{fill:amber,font:{bold:true,color:"#7F6000"}}});

title(ap,"v3 application profile","Concrete format decisions derived from official rules, stable conventions and the user-approved v3 plan.","D"); ap.getRange("A4:D4").values=[["Element","v3 setting","Basis","Files"]]; ap.getRange("A5:D14").values=applications; header(ap.getRange("A4:D4")); body(ap.getRange("A5:D14")); ap.getRange("A:D").format.columnWidth=28; ap.getRange("B:B").format.columnWidth=48; ap.getRange("C:D").format.columnWidth=32;

await exportAndRender(formatWb,"EAAI_12_Format_Comparison_v3.xlsx",[["Summary","A1:F14"],["Paper Format","A1:N16"],["Rules","A1:F15"],["Application","A1:D14"]]);

const terms = [
  ["PPO–Pointer","PPO–Pointer","近端策略优化–Pointer策略模型","Keep model name; explain once at first use","PPO-Pointer, PPO + Pointer","Locked","Abstract / 摘要"],
  ["A2C–Pointer","A2C–Pointer","优势演员–评论家–Pointer策略模型","Keep model name; explain once","A2C-Pointer","Locked","Abstract / 摘要"],
  ["Flat-MLP PPO","Flat-MLP PPO","固定槽多层感知机PPO","Do not call traditional PPO alone","PPO-MLP, ppo_mlp","Locked","Abstract / 摘要"],
  ["digital surface model","DSM","数字表面模型","Use DSM after first definition","数字高程模型 (unless source is DEM)","Locked","Abstract / 摘要"],
  ["digital elevation model","DEM","数字高程模型","Retain Copernicus DEM GLO-30 product name","DSM","Locked","Methods / 方法"],
  ["safe weighted coverage","SWC","安全加权覆盖率","Priority-weighted coverage set to zero on unsafe failure","安全覆盖率","Locked","Glossary / 术语表"],
  ["return-aware feasibility mask","—","返航感知可行性掩码","Composite mechanism; do not claim isolated submask effects","安全掩码","Locked","Glossary / 术语表"],
  ["priority-weighted coverage","—","优先级加权覆盖率","Distinguish from SWC when safety gate is absent","加权覆盖率","Locked","Methods / 方法"],
  ["safe return","—","安全返航","Avoid certification wording","保证安全","Locked","Abstract / 摘要"],
  ["zero-shot simulation transfer","—","零样本仿真迁移","Never translate as real-world transfer","零样本实地迁移","Locked","Abstract / 摘要"],
  ["unseen procedural maps","—","未见程序化地图","Within trained node-count range","未知地图","Locked","Methods / 方法"],
  ["known domain shift","—","已知域偏移","Planner receives shift-consistent observations","已知扰动","Locked","Methods / 方法"],
  ["hidden model/perception mismatch","—","隐藏模型/感知失配","Planning and execution truth differ","未知域偏移","Locked","Methods / 方法"],
  ["return reserve","—","返航储备","Ablated as a composite reserve mechanism","返航余量子掩码","Locked","Methods / 方法"],
  ["map-level inference","—","地图级推断","Maps are independent units","任务级推断","Locked","Statistics / 统计"],
  ["bootstrap interval","—","bootstrap区间","Keep method name in English; retain level and unit","置信区间 (without method)","Locked","Statistics / 统计"],
  ["Holm-adjusted p value","—","Holm校正p值","Preserve family and multiplicity context","校正后显著","Locked","Statistics / 统计"],
  ["empirical cumulative distribution","ECDF","经验累积分布","Expand at first use if abbreviation appears","累计概率图","Locked","Results / 结果"],
  ["online planning time","—","在线规划时间","Protocol-specific; not a platform guarantee","推理时间","Locked","Results / 结果"],
  ["sample efficiency","—","样本效率","Post-hoc training dimension","数据效率","Locked","Results / 结果"],
  ["training stability","—","训练稳定性","Post-hoc dimension; not final-route endpoint","策略稳定性","Locked","Results / 结果"],
  ["ablation","—","消融","Four frozen ablations only","删减实验","Locked","Results / 结果"],
  ["observation","—","观察结果","Directly measured","结论","Locked","Discussion / 讨论"],
  ["interpretation","—","解释","Bounded explanation consistent with evidence","机制证明","Locked","Discussion / 讨论"],
  ["mechanistic possibility","—","机制可能性","Not causally isolated","作用机制","Locked","Discussion / 讨论"],
  ["simulation-only","—","仅仿真","No flight validation","数字孪生验证","Locked","Limitations / 局限"],
  ["depot","—","基地","Launch and return location","巢站 (unless quoting exemplar)","Locked","Methods / 方法"],
  ["inspection point","—","巡检点","Fixed priority-labelled point","航点","Locked","Throughout / 全文"],
  ["Source Data","—","Source Data","Keep official package label in English","源数据图","Locked","Declarations / 声明"],
  ["Original Research","—","Original Research","Keep EAAI article-type label in English","研究论文类型猜测","Locked","Title page / 标题页"],
];

const termWb=Workbook.create(); const ts=termWb.worksheets.add("Summary"); const tr=termWb.worksheets.add("Terms"); title(ts,"English–Chinese terminology glossary v3","Locked forms preserve model identity, evidence boundaries, statistical meaning and figure/document consistency.","E"); ts.getRange("A4:B8").values=[["Metric","Value"],["Terms",null],["Locked",null],["Model-name terms",null],["Prepared","2026-08-09"]]; ts.getRange("B5").formulas=[["=COUNTA('Terms'!A5:A34)"]]; ts.getRange("B6").formulas=[["=COUNTIF('Terms'!F5:F34,\"Locked\")"]]; ts.getRange("B7").formulas=[["=COUNTA('Terms'!A5:A7)"]]; header(ts.getRange("A4:B4")); body(ts.getRange("A5:B8")); ts.getRange("A5:A8").format={fill:green,font:{bold:true,color:ink},borders:{preset:"all",style:"thin",color:grid}}; ts.getRange("A:B").format.columnWidth=34;
title(tr,"Locked bilingual terminology","Reference titles and bibliographic metadata remain in their original English and are not included as translation terms.","G"); tr.getRange("A4:G4").values=[["English","Abbreviation","中文规范译法","Use rule","Forbidden / avoid","Status","First-use location"]]; tr.getRange("A5:G34").values=terms; header(tr.getRange("A4:G4")); body(tr.getRange("A5:G34")); tr.freezePanes.freezeRows(4); tr.getRange("A:G").format.columnWidth=24; tr.getRange("C:E").format.columnWidth=38; tr.getRange("G:G").format.columnWidth=24; tr.getRange("F5:F34").conditionalFormats.add("containsText",{text:"Locked",format:{fill:green,font:{bold:true,color:"#006100"}}});
await exportAndRender(termWb,"EN_ZH_Terminology_Glossary_v3.xlsx",[["Summary","A1:E10"],["Terms","A1:G34"]]);

console.log(JSON.stringify({workbooks:2,previews:6},null,2));
