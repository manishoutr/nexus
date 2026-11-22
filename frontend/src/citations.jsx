// Turns "[1]" or "[file.txt, page 2]" markers in an answer into clickable chips.
export function splitAnswer(answer, citations) {
  const parts = [];
  let last = 0;
  const re = /\[([^\]]+)\]/g;
  let m;
  while ((m = re.exec(answer))) {
    const inner = m[1].trim();
    let idx = -1;
    if (/^\d+$/.test(inner)) idx = Number(inner) - 1;
    else {
      const name = inner.split(",")[0].trim().toLowerCase();
      idx = citations.findIndex((c) => c.document_name.toLowerCase() === name);
    }
    if (idx < 0 || !citations[idx]) continue;
    if (m.index > last) parts.push({ text: answer.slice(last, m.index) });
    parts.push({ chip: idx });
    last = m.index + m[0].length;
  }
  if (last < answer.length) parts.push({ text: answer.slice(last) });
  return parts;
}
