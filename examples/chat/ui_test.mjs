/**
 * Tests for index.html, run with `node examples/chat/ui_test.mjs`.
 *
 * The page has no build step and no framework, so this stubs the handful of DOM
 * calls it makes and exercises the script directly. It exists because a slip in
 * rendering the file panel once swallowed the approval gate, which looks exactly
 * like a hung agent: reasoning appears, then nothing, and there is no way to
 * allow the call that is waiting.
 */
import { readFileSync } from "node:fs";

function node(tag) {
  return {
    tag, className: "", textContent: "", children: [], open: false, onclick: null,
    append(...kids) { this.children.push(...kids); },
    replaceChildren(...kids) { this.children = kids; },
    querySelector(sel) { return find(this, sel); },
    querySelectorAll() { return []; },
    addEventListener() {},
    get lastChild() { return this.children.at(-1); },
    remove() {},
  };
}
function find(root, sel) {
  const want = sel.replace(/[.#]/g, "").split(":")[0];
  for (const kid of root.children ?? []) {
    if (kid.tag === want || (kid.className ?? "").split(" ").includes(want)) return kid;
    const deeper = find(kid, sel);
    if (deeper) return deeper;
  }
  return null;
}
function walk(root, out = []) {
  for (const kid of root.children ?? []) { out.push(kid); walk(kid, out); }
  return out;
}

const ids = {};
for (const id of ["log", "files", "tools", "input", "send", "meta", "composer", "reset", "viewer", "viewer-name", "viewer-body"]) {
  ids[id] = node("div");
}
globalThis.document = { getElementById: (id) => ids[id], createElement: node };
globalThis.window = { addEventListener() {}, __model: "test-model" };
globalThis.fetch = async () => ({ ok: true, json: async () => ({ model: "m", workspace: "/w", tools: [], paused: [], files: [] }) });
globalThis.TextDecoder = class { decode() { return ""; } };

const html = readFileSync("examples/chat/index.html", "utf8");
const script = html.slice(html.lastIndexOf("<script>") + 8, html.lastIndexOf("</script>"));
const run = new Function(`${script}\nreturn { showFiles, render, gate };`);
const { showFiles, render } = run();

let failures = 0;
function check(name, fn) {
  try { fn(); console.log(`  ok   ${name}`); }
  catch (exc) { failures++; console.log(`  FAIL ${name}: ${exc.message}`); }
}

check("an empty workspace renders without throwing", () => {
  showFiles([]);
  if (!ids.files.children.length) throw new Error("nothing rendered");
});

check("a paused run puts up the approval gate", () => {
  const holder = node("div");
  render(
    {
      type: "done", stop_reason: "paused", usage: { total: 10 }, files: [],
      paused: [{ call_id: "c1", name: "write_file", arguments: { path: "a.py" }, question: "Run write_file?" }],
    },
    { holder, model: "m" }
  );
  const gates = walk(holder).filter((n) => (n.className ?? "").includes("gate"));
  if (!gates.length) throw new Error("no gate rendered");
  const buttons = walk(holder).filter((n) => n.tag === "button");
  if (buttons.length < 2) throw new Error(`expected approve+reject, got ${buttons.length}`);
});

check("a paused run with files still gates", () => {
  const holder = node("div");
  render(
    { type: "done", stop_reason: "paused", usage: { total: 1 }, files: [{ path: "a.py", bytes: 3 }],
      paused: [{ call_id: "c", name: "edit_file", arguments: {}, question: "?" }] },
    { holder, model: "m" }
  );
  if (!walk(holder).some((n) => (n.className ?? "").includes("gate"))) throw new Error("no gate");
});

console.log(failures ? `\n${failures} failing` : "\nall ok");
process.exit(failures ? 1 : 0);
