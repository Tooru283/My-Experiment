const pptxgen = require('pptxgenjs');
const path = require('path');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'moca';
pptx.company = 'Controlled Navigation Harness';
pptx.subject = 'VLN-CE experiment progress report';
pptx.title = '让导航智能体到达后停得住';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: 'Arial Unicode MS',
  bodyFontFace: 'Arial Unicode MS',
  lang: 'zh-CN',
};
pptx.defineSlideMaster({
  title: 'MASTER',
  background: { color: 'F7F8FA' },
  objects: [
    { rect: { x: 0, y: 0, w: 13.333, h: 0.08, fill: { color: '2A6F97' }, line: { color: '2A6F97' } } },
    { text: { text: '室内视觉语言导航实验进展', options: { x: 0.62, y: 7.12, w: 4.0, h: 0.18, fontFace: 'Arial Unicode MS', fontSize: 8.5, color: '77818B', margin: 0 } } },
    { text: { text: '2026-07-13', options: { x: 11.45, y: 7.12, w: 1.2, h: 0.18, fontFace: 'Aptos', fontSize: 8.5, color: '77818B', align: 'right', margin: 0 } } },
  ],
  slideNumber: { x: 12.73, y: 7.08, w: 0.24, h: 0.2, fontFace: 'Aptos', fontSize: 8.5, color: '77818B', align: 'right' },
});

const C = {
  ink: '17212B',
  text: '34414D',
  muted: '6E7984',
  line: 'D7DCE1',
  blue: '2A6F97',
  blueLight: 'DCEBF3',
  green: '2E7D62',
  greenLight: 'DCEBE5',
  red: 'B44E4E',
  redLight: 'F3DEDE',
  amber: 'A66B1F',
  amberLight: 'F3E8D5',
  white: 'FFFFFF',
  bg: 'F7F8FA',
};

const S = pptx.ShapeType;

function addTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.72, y: 0.42, w: 11.9, h: 0.48,
    fontFace: 'Arial Unicode MS', fontSize: 25, bold: true,
    color: C.ink, margin: 0, breakLine: false,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.74, y: 0.97, w: 11.7, h: 0.28,
      fontFace: 'Arial Unicode MS', fontSize: 11.5,
      color: C.muted, margin: 0,
    });
  }
}

function addSectionLabel(slide, text, x, y, w, color = C.blue) {
  slide.addText(text, {
    x, y, w, h: 0.26, fontFace: 'Arial Unicode MS', fontSize: 11,
    bold: true, color, margin: 0,
  });
}

function addBullet(slide, text, x, y, w, opts = {}) {
  slide.addShape(S.ellipse, {
    x, y: y + 0.12, w: 0.08, h: 0.08,
    fill: { color: opts.color || C.blue }, line: { color: opts.color || C.blue },
  });
  slide.addText(text, {
    x: x + 0.22, y, w: w - 0.22, h: opts.h || 0.42,
    fontFace: 'Arial Unicode MS', fontSize: opts.fontSize || 15,
    color: opts.textColor || C.text, margin: 0,
    bold: opts.bold || false, valign: 'mid',
  });
}

function addPill(slide, text, x, y, w, fill, color) {
  slide.addShape(S.roundRect, {
    x, y, w, h: 0.32, rectRadius: 0.05,
    fill: { color: fill }, line: { color: fill },
  });
  slide.addText(text, {
    x, y: y + 0.01, w, h: 0.26, align: 'center', valign: 'mid',
    fontFace: 'Arial Unicode MS', fontSize: 9.5, bold: true,
    color, margin: 0,
  });
}

function addArrow(slide, x, y, w, color = C.line) {
  slide.addShape(S.line, {
    x, y, w, h: 0,
    line: { color, width: 1.8, endArrowType: 'triangle' },
  });
}

function addAcademicTable(slide, columns, rows, opts = {}) {
  const x = opts.x || 0.84;
  const y = opts.y || 1.58;
  const headerH = opts.headerH || 0.52;
  const rowH = opts.rowH || 0.52;
  const width = columns.reduce((sum, col) => sum + col.w, 0);

  slide.addShape(S.line, {
    x, y, w: width, h: 0,
    line: { color: C.ink, width: 1.5 },
  });
  slide.addShape(S.rect, {
    x, y: y + 0.02, w: width, h: headerH - 0.03,
    fill: { color: 'EEF2F4' }, line: { color: 'EEF2F4' },
  });

  let cx = x;
  columns.forEach((col) => {
    slide.addText(col.title, {
      x: cx + 0.08, y: y + 0.13, w: col.w - 0.16, h: 0.24,
      fontFace: 'Arial Unicode MS', fontSize: opts.headerFontSize || 11.5,
      bold: true, color: C.ink, align: col.align || 'left',
      valign: 'mid', margin: 0,
    });
    cx += col.w;
  });
  slide.addShape(S.line, {
    x, y: y + headerH, w: width, h: 0,
    line: { color: C.ink, width: 0.9 },
  });

  rows.forEach((row, rowIndex) => {
    const rowY = y + headerH + rowIndex * rowH;
    if (row.fill) {
      slide.addShape(S.rect, {
        x, y: rowY + 0.01, w: width, h: rowH - 0.02,
        fill: { color: row.fill }, line: { color: row.fill },
      });
    }
    if (row.groupStart && rowIndex > 0) {
      slide.addShape(S.line, {
        x, y: rowY, w: width, h: 0,
        line: { color: 'AEB7BF', width: 0.8 },
      });
    }

    let cellX = x;
    row.cells.forEach((rawCell, cellIndex) => {
      const cell = typeof rawCell === 'string' ? { text: rawCell } : rawCell;
      const col = columns[cellIndex];
      slide.addText(cell.text, {
        x: cellX + 0.08, y: rowY + 0.11, w: col.w - 0.16, h: rowH - 0.18,
        fontFace: cell.fontFace || 'Arial Unicode MS',
        fontSize: cell.fontSize || opts.fontSize || 11.5,
        bold: cell.bold || false,
        color: cell.color || C.text,
        align: cell.align || col.align || 'left',
        valign: 'mid', margin: 0,
      });
      cellX += col.w;
    });
    if (rowIndex < rows.length - 1 && opts.rowLines) {
      slide.addShape(S.line, {
        x, y: rowY + rowH, w: width, h: 0,
        line: { color: C.line, width: 0.45 },
      });
    }
  });

  slide.addShape(S.line, {
    x, y: y + headerH + rows.length * rowH, w: width, h: 0,
    line: { color: C.ink, width: 1.5 },
  });
}

// Slide 1: title and problem sketch
{
  const slide = pptx.addSlide('MASTER');
  slide.background = { color: C.bg };
  slide.addText('让导航智能体到达后\n“停得住”', {
    x: 0.76, y: 1.18, w: 6.2, h: 1.45,
    fontFace: 'Arial Unicode MS', fontSize: 34, bold: true,
    color: C.ink, margin: 0, breakLine: false, valign: 'mid',
  });
  slide.addText('核心科学问题', {
    x: 0.78, y: 3.03, w: 1.3, h: 0.28,
    fontFace: 'Arial Unicode MS', fontSize: 11, bold: true,
    color: C.blue, margin: 0,
  });
  slide.addText('智能体已经进入目标范围，却没有及时停止，最后又走了出去。', {
    x: 0.78, y: 3.38, w: 5.65, h: 1.05,
    fontFace: 'Arial Unicode MS', fontSize: 20, bold: true,
    color: C.text, margin: 0, breakLine: false, valign: 'mid',
  });
  slide.addShape(S.line, {
    x: 0.78, y: 4.75, w: 4.75, h: 0,
    line: { color: C.line, width: 1.2 },
  });
  slide.addText('从“曾经到过”转化为“最终停对”', {
    x: 0.78, y: 4.98, w: 5.6, h: 0.42,
    fontFace: 'Arial Unicode MS', fontSize: 15.5, color: C.green,
    bold: true, margin: 0,
  });

  // Simple route diagram
  slide.addShape(S.ellipse, {
    x: 8.48, y: 1.48, w: 3.25, h: 3.25,
    fill: { color: C.greenLight, transparency: 22 },
    line: { color: C.green, width: 2, dash: 'dash' },
  });
  slide.addShape(S.ellipse, {
    x: 9.72, y: 2.71, w: 0.76, h: 0.76,
    fill: { color: C.green }, line: { color: C.green },
  });
  slide.addText('目标', {
    x: 9.82, y: 2.91, w: 0.55, h: 0.25,
    fontFace: 'Arial Unicode MS', fontSize: 11, bold: true,
    color: C.white, align: 'center', margin: 0,
  });
  slide.addText('目标范围（3m）', {
    x: 8.82, y: 1.18, w: 2.6, h: 0.28,
    fontFace: 'Arial Unicode MS', fontSize: 12, bold: true,
    color: C.green, align: 'center', margin: 0,
  });
  const pathIn = [
    [7.25, 5.92, 1.02, -0.82],
    [8.23, 5.12, 0.84, -0.68],
    [9.03, 4.48, 0.72, -0.72],
    [9.71, 3.79, 0.35, -0.52],
  ];
  pathIn.forEach(([x, y, w, h]) => slide.addShape(S.line, {
    x, y, w, h, line: { color: C.blue, width: 4, beginArrowType: 'none', endArrowType: 'none' },
  }));
  slide.addShape(S.line, {
    x: 10.06, y: 3.27, w: 1.76, h: -1.35,
    line: { color: C.red, width: 4, dash: 'dash', endArrowType: 'triangle' },
  });
  slide.addShape(S.ellipse, {
    x: 7.06, y: 5.76, w: 0.33, h: 0.33,
    fill: { color: C.blue }, line: { color: C.white, width: 1.2 },
  });
  slide.addText('起点', {
    x: 6.78, y: 6.15, w: 0.9, h: 0.25,
    fontFace: 'Arial Unicode MS', fontSize: 10.5, color: C.muted, align: 'center', margin: 0,
  });
  addPill(slide, '已经进入范围', 8.19, 4.82, 1.55, C.blueLight, C.blue);
  addPill(slide, '继续走出', 11.08, 1.28, 1.24, C.redLight, C.red);
  slide.addNotes('老师可以把这个任务理解成虚拟机器人找路。它根据一句文字指令在室内移动，最终要停在目标附近。我现在重点解决的不是单纯“能不能找到目标”，而是它明明已经进入成功范围，却没有意识到应该停下，最后又走了出去。');
}

// Slide 2: size of the problem
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, '到过目标，不等于成功完成任务', '原始 4B 基线：100 个导航任务');

  const boxes = [
    { x: 0.92, w: 2.52, fill: 'E8ECEF', line: 'B9C1C8', num: '100', label: '全部任务', color: C.ink },
    { x: 5.26, w: 2.52, fill: C.blueLight, line: C.blue, num: '25', label: '曾进入目标范围', color: C.blue },
    { x: 9.62, w: 2.52, fill: C.greenLight, line: C.green, num: '16', label: '最终成功停下', color: C.green },
  ];
  boxes.forEach((b) => {
    slide.addShape(S.roundRect, {
      x: b.x, y: 2.0, w: b.w, h: 2.12,
      fill: { color: b.fill }, line: { color: b.line, width: 1.5 },
      radius: 0.06,
    });
    slide.addText(b.num, {
      x: b.x, y: 2.35, w: b.w, h: 0.75,
      fontFace: 'Aptos Display', fontSize: 38, bold: true,
      color: b.color, align: 'center', margin: 0,
    });
    slide.addText(b.label, {
      x: b.x + 0.15, y: 3.25, w: b.w - 0.3, h: 0.38,
      fontFace: 'Arial Unicode MS', fontSize: 15, bold: true,
      color: C.text, align: 'center', margin: 0,
    });
  });
  addArrow(slide, 3.64, 3.04, 1.25, C.muted);
  addArrow(slide, 7.98, 3.04, 1.28, C.muted);

  slide.addShape(S.line, {
    x: 7.05, y: 4.44, w: 3.55, h: 0,
    line: { color: C.red, width: 2.2, beginArrowType: 'triangle', endArrowType: 'triangle' },
  });
  slide.addText('9 次已经到手的机会又丢掉了', {
    x: 7.22, y: 4.64, w: 3.2, h: 0.35,
    fontFace: 'Arial Unicode MS', fontSize: 14, bold: true,
    color: C.red, align: 'center', margin: 0,
  });

  slide.addShape(S.rect, {
    x: 1.02, y: 5.46, w: 11.15, h: 0.9,
    fill: { color: 'EEF2F4' }, line: { color: 'EEF2F4' },
  });
  slide.addText('到过率 25%', {
    x: 1.35, y: 5.72, w: 2.2, h: 0.34,
    fontFace: 'Arial Unicode MS', fontSize: 18, bold: true, color: C.blue, margin: 0,
  });
  slide.addText('最终成功率 16%', {
    x: 4.47, y: 5.72, w: 2.5, h: 0.34,
    fontFace: 'Arial Unicode MS', fontSize: 18, bold: true, color: C.green, margin: 0,
  });
  slide.addText('到达后的成功转化率 64%', {
    x: 8.18, y: 5.72, w: 3.42, h: 0.34,
    fontFace: 'Arial Unicode MS', fontSize: 18, bold: true, color: C.ink, margin: 0,
  });
  slide.addNotes('在这个任务里，最终停在目标三米范围内才算成功。原始4B基线的100个任务中，有25个任务曾经进入目标范围，但最终只有16个任务成功停下。也就是说，有9次已经走到了，却因为停止判断不可靠又走了出去。');
}

// Slide 3: workload table
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, '近一个月完成的实验与研究产出', '统计范围：2026-06-10 至 2026-07-05；数字来自当前项目目录');
  const workloadColumns = [
    { title: '类别', w: 1.35 },
    { title: '统计项', w: 3.15 },
    { title: '数量', w: 2.05, align: 'center' },
    { title: '口径说明', w: 5.10 },
  ];
  const workloadRows = [
    { cells: [{ text: '实验评测', bold: true, color: C.blue }, '完成评测', { text: '48 轮', bold: true, color: C.blue, align: 'center' }, '均产出汇总指标'] },
    { cells: ['', '完整 100 集评测', { text: '17 轮', bold: true, color: C.blue, align: 'center' }, '100 个任务的完整对照'] },
    { cells: ['', '累计评测任务', { text: '1972 个', bold: true, color: C.blue, align: 'center' }, '按各轮配置规模合计'] },
    { cells: ['', '运行日志', { text: '79 份', bold: true, color: C.blue, align: 'center' }, '记录环境、配置与运行状态'] },
    { groupStart: true, cells: [{ text: '数据沉淀', bold: true, color: C.green }, '逐 episode trace', { text: '2059 份', bold: true, color: C.green, align: 'center' }, '逐任务结构化追踪'] },
    { cells: ['', '结构化决策 trace', { text: '约 64.5 万行', bold: true, color: C.green, align: 'center' }, '感知、选择与停止事件'] },
    { cells: ['', '导航事件日志', { text: '约 59.6 万行', bold: true, color: C.green, align: 'center' }, '动作、位置与状态变化'] },
    { cells: ['', '日志总体积', { text: '1.8 GB', bold: true, color: C.green, align: 'center' }, '当前 logs 目录统计'] },
    { groupStart: true, cells: [{ text: '工程产出', bold: true, color: C.amber }, '导航扩展模块', { text: '24 个', bold: true, color: C.amber, align: 'center' }, '约 4990 行扩展代码'] },
    { groupStart: true, cells: [{ text: '研究产出', bold: true, color: C.red }, '项目文档', { text: '55 篇', bold: true, color: C.red, align: 'center' }, '含 9 篇记录、3 份报告、10 张分析图'] },
  ];
  addAcademicTable(slide, workloadColumns, workloadRows, {
    x: 0.84, y: 1.48, rowH: 0.48, headerH: 0.50,
    fontSize: 11.2, headerFontSize: 11.4, rowLines: true,
  });
  slide.addNotes('这段时间的工作量不仅是修改代码，还包括反复运行、对照和复盘。近一个月完成了48轮正式评测，其中17轮是完整的100集评测，累计接近2000个导航任务。项目保留了逐任务和逐步骤记录，因此可以追踪智能体什么时候进入目标范围、为什么没有停、最后走向了哪里。');
}

// Slide 4: methods
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, '现在的技术路线：给“该不该停”增加外部证据', '不只依赖模型自报，而是让停止判断可观察、可复盘、可消融');

  addSectionLabel(slide, '已经接入实验系统', 0.82, 1.54, 2.5, C.blue);
  addSectionLabel(slide, '正在做单开关验证', 7.05, 1.54, 2.5, C.green);
  slide.addShape(S.line, { x: 6.65, y: 1.55, w: 0, h: 4.2, line: { color: C.line, width: 1.2 } });

  const completed = [
    ['受控实验框架', '记录每一步看到了什么、选了哪里、为什么停或不停'],
    ['视觉停止验证', '模型提出停止后，再检查目标证据是否充分'],
    ['主动停止与恢复', '接近目标时主动触发判断，拒停后继续探索或换路'],
  ];
  const testing = [
    ['深度停止判断', '目标明显过远时，不允许轻易停止'],
    ['路线状态一致性', '按指令顺序维护“当前走到哪一步”'],
    ['回溯与负目标记忆', '走错后退回，错误位置不再反复尝试'],
  ];

  function addMethodList(items, x, y, w, accent, fill) {
    items.forEach((item, i) => {
      const yy = y + i * 1.15;
      slide.addShape(S.ellipse, {
        x, y: yy + 0.04, w: 0.36, h: 0.36,
        fill: { color: fill }, line: { color: accent, width: 1.2 },
      });
      slide.addText(String(i + 1), {
        x, y: yy + 0.09, w: 0.36, h: 0.2,
        fontFace: 'Aptos', fontSize: 10.5, bold: true, color: accent,
        align: 'center', margin: 0,
      });
      slide.addText(item[0], {
        x: x + 0.55, y: yy, w: w - 0.55, h: 0.3,
        fontFace: 'Arial Unicode MS', fontSize: 15.5, bold: true, color: C.ink, margin: 0,
      });
      slide.addText(item[1], {
        x: x + 0.55, y: yy + 0.39, w: w - 0.55, h: 0.48,
        fontFace: 'Arial Unicode MS', fontSize: 11.8, color: C.muted, margin: 0, breakLine: false,
      });
    });
  }
  addMethodList(completed, 0.84, 2.02, 5.35, C.blue, C.blueLight);
  addMethodList(testing, 7.08, 2.02, 5.25, C.green, C.greenLight);

  slide.addShape(S.rect, {
    x: 1.04, y: 5.85, w: 11.08, h: 0.62,
    fill: { color: 'EEF2F4' }, line: { color: 'EEF2F4' },
  });
  const stages = [
    ['看见', '视觉与深度证据'],
    ['判断', '路线状态与停止核验'],
    ['停止', '主动把握成功窗口'],
    ['恢复', '回溯与错误记忆'],
  ];
  stages.forEach((st, i) => {
    const x = 1.28 + i * 2.73;
    slide.addText(st[0], {
      x, y: 6.02, w: 0.72, h: 0.24,
      fontFace: 'Arial Unicode MS', fontSize: 12.5, bold: true, color: i < 2 ? C.blue : C.green, margin: 0,
    });
    slide.addText(st[1], {
      x: x + 0.72, y: 6.03, w: 1.55, h: 0.22,
      fontFace: 'Arial Unicode MS', fontSize: 10.2, color: C.muted, margin: 0,
    });
    if (i < stages.length - 1) addArrow(slide, x + 2.23, 6.15, 0.36, C.line);
  });
  slide.addNotes('现在的实验围绕同一个问题设计：不能只听模型说“我到了”。系统要结合视觉证据、深度信息和路线进度进行判断。已经接入的部分包括受控实验框架、视觉停止验证、主动停止和失败恢复；现在正在做深度判断、路线状态、回溯和负目标记忆的单开关验证。');
}

// Slide 5: results
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, '阶段结果：主要提升来自“到达后真正停下”', 'SR：最终成功率；OSR：过程中曾进入目标范围的比例；↑ 越高越好，↓ 越低越好');

  const resultColumns = [
    { title: '实验配置', w: 2.35 },
    { title: 'SR ↑', w: 1.25, align: 'center' },
    { title: 'OSR ↑', w: 1.25, align: 'center' },
    { title: 'OSR→SR ↑', w: 1.75, align: 'center' },
    { title: '未转化数 ↓', w: 1.70, align: 'center' },
    { title: '结果说明', w: 3.35 },
  ];
  const resultRows = [
    {
      cells: [
        { text: '原始 4B 基线', bold: true },
        { text: '16%', align: 'center' },
        { text: '25%', align: 'center' },
        { text: '64.0%', align: 'center' },
        { text: '9', align: 'center' },
        '到达后仍有较多机会丢失',
      ],
    },
    {
      fill: 'E8F2ED',
      cells: [
        { text: '4B 阶段最好', bold: true, color: C.green },
        { text: '24%', bold: true, color: C.green, align: 'center' },
        { text: '26%', bold: true, color: C.green, align: 'center' },
        { text: '92.3%', bold: true, color: C.green, align: 'center' },
        { text: '2', bold: true, color: C.green, align: 'center' },
        { text: '大部分到达机会转化为成功', bold: true, color: C.green },
      ],
    },
    {
      groupStart: true,
      cells: [
        { text: '变化（阶段最好−基线）', bold: true, color: C.blue },
        { text: '+8 pp', bold: true, color: C.blue, align: 'center' },
        { text: '+1 pp', bold: true, color: C.blue, align: 'center' },
        { text: '+28.3 pp', bold: true, color: C.blue, align: 'center' },
        { text: '−7', bold: true, color: C.red, align: 'center' },
        { text: '主要收益来自停止转化', bold: true, color: C.blue },
      ],
    },
  ];
  addAcademicTable(slide, resultColumns, resultRows, {
    x: 0.84, y: 1.78, rowH: 0.78, headerH: 0.58,
    fontSize: 12.2, headerFontSize: 11.8, rowLines: true,
  });

  slide.addShape(S.line, {
    x: 1.02, y: 5.05, w: 11.05, h: 0,
    line: { color: C.line, width: 0.9 },
  });
  slide.addText('读表结论', {
    x: 1.04, y: 5.32, w: 1.05, h: 0.28,
    fontFace: 'Arial Unicode MS', fontSize: 11, bold: true, color: C.muted, margin: 0,
  });
  slide.addText('OSR 仅提高 1 个百分点，但 SR 提高 8 个百分点：性能提升主要来自把“已经到达”的机会真正转化为成功。', {
    x: 2.10, y: 5.24, w: 9.7, h: 0.58,
    fontFace: 'Arial Unicode MS', fontSize: 16, bold: true, color: C.ink,
    margin: 0, valign: 'mid',
  });
  slide.addText('未转化任务从 9 个降至 2 个，正好对应本实验关注的“进入目标范围后又走出来”问题。', {
    x: 2.10, y: 5.94, w: 9.7, h: 0.38,
    fontFace: 'Arial Unicode MS', fontSize: 12.5, color: C.red,
    bold: true, margin: 0,
  });
  slide.addNotes('阶段结果说明，这条技术路线确实对准了问题。系统曾经到达目标的能力变化不大，从25%到26%，但最终成功率从16%提高到了24%。提升主要来自把已经到达的机会真正转化成成功，丢失机会从9个减少到2个。');
}

// Slide 6: now and next
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, '现在在做什么，接下来准备做什么', '保持单开关、可归因的实验节奏');

  const cols = [
    {
      x: 0.82, title: '已经完成', color: C.blue, fill: C.blueLight,
      items: ['跑通完整实验链路', '建立可复盘的受控框架', '完成多轮 4B / 9B 对照', '定位“到过但没停住”问题'],
    },
    {
      x: 4.51, title: '正在做', color: C.green, fill: C.greenLight,
      items: ['深度停止判断', '路线状态一致性', '可观测回溯', '负目标位置记忆'],
    },
    {
      x: 8.20, title: '准备做', color: C.amber, fill: C.amberLight,
      items: ['逐项完成单开关 A/B', '合并有效机制后跑 100 集', '筛选更合适的模型底座', '后续训练选路与停止模块'],
    },
  ];
  cols.forEach((col, idx) => {
    slide.addShape(S.rect, {
      x: col.x, y: 1.62, w: 3.22, h: 0.62,
      fill: { color: col.fill }, line: { color: col.fill },
    });
    slide.addText(col.title, {
      x: col.x + 0.2, y: 1.81, w: 1.7, h: 0.27,
      fontFace: 'Arial Unicode MS', fontSize: 15, bold: true, color: col.color, margin: 0,
    });
    addPill(slide, String(idx + 1), col.x + 2.60, 1.76, 0.34, C.white, col.color);
    col.items.forEach((item, i) => addBullet(slide, item, col.x + 0.16, 2.65 + i * 0.76, 2.9, {
      color: col.color, fontSize: 13.2, h: 0.46,
    }));
  });
  addArrow(slide, 4.12, 1.94, 0.26, C.line);
  addArrow(slide, 7.81, 1.94, 0.26, C.line);

  slide.addShape(S.rect, {
    x: 0, y: 6.18, w: 13.333, h: 0.72,
    fill: { color: C.ink }, line: { color: C.ink },
  });
  slide.addText('不是只让智能体“走到过目标”，而是让它在到达时判断出来，并且可靠地停下。', {
    x: 1.0, y: 6.40, w: 11.35, h: 0.3,
    fontFace: 'Arial Unicode MS', fontSize: 16.5, bold: true,
    color: C.white, align: 'center', margin: 0,
  });
  slide.addNotes('下一步继续按照单开关实验推进，一次只验证一个机制，确保能够说明提升来自哪里。有效机制再组合起来进行完整100集评测。后续还会筛选更合适的模型底座，并尝试训练选路与停止模块。我的工作不是只让智能体走到过目标，而是让它在真正到达时能够判断出来，并可靠地停下。');
}

const outputPath = path.resolve(__dirname, '..', 'outputs', '导航实验进展汇报-5min-20260713.pptx');
pptx.writeFile({ fileName: outputPath });
