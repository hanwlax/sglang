#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const root = process.cwd();
const commit = process.argv[2] || "";
const scan = JSON.parse(
  fs.readFileSync(path.join(root, ".understand-anything/intermediate/scan-result.json"), "utf8")
);

const now = new Date().toISOString();

function nodeType(file) {
  const p = file.path;
  if (file.fileCategory === "config") return "config";
  if (file.fileCategory === "docs") return "document";
  if (file.fileCategory === "infra") {
    if (p.startsWith(".github/workflows/") || p.includes("gitlab-ci") || p.includes("Jenkinsfile")) return "pipeline";
    if (p.endsWith(".tf") || p.endsWith(".tfvars") || p.includes("k8s") || p.includes("kubernetes")) return "resource";
    return "service";
  }
  if (file.fileCategory === "data") {
    if (p.endsWith(".graphql") || p.endsWith(".gql") || p.endsWith(".proto") || p.endsWith(".prisma")) return "schema";
    if (p.endsWith(".sql")) return "table";
    return "schema";
  }
  return "file";
}

function idFor(file) {
  return `${nodeType(file)}:${file.path}`;
}

function complexity(file) {
  if (file.sizeLines > 200) return "complex";
  if (file.sizeLines >= 50) return "moderate";
  return "simple";
}

function tagsFor(file, type) {
  const p = file.path;
  const tags = new Set();
  tags.add(file.fileCategory);
  tags.add(file.language);
  if (p.includes("/test") || p.startsWith("test/") || /test_|_test|\.spec\.|\.test\./.test(p)) tags.add("test");
  if (p.includes("server_args")) tags.add("configuration");
  if (p.includes("scheduler")) tags.add("scheduler");
  if (p.includes("model_runner")) tags.add("model-execution");
  if (p.includes("mem_cache") || p.includes("cache")) tags.add("cache");
  if (p.includes("Dockerfile") || p.startsWith("docker/")) tags.add("deployment");
  if (type === "pipeline") tags.add("ci-cd");
  if (type === "document") tags.add("documentation");
  if (type === "config") tags.add("configuration");
  return Array.from(tags).slice(0, 5);
}

function summaryFor(file, type) {
  const p = file.path;
  if (type === "document") return `文档文件，记录 ${p} 相关的使用、设计或维护信息。`;
  if (type === "config") return `配置文件，控制 ${p} 对应的构建、测试、运行或工具行为。`;
  if (type === "pipeline") return `CI/CD 流水线配置，定义自动化检查、构建或发布流程。`;
  if (type === "service" || type === "resource") return `部署或基础设施文件，描述 ${p} 相关的镜像、服务或运行环境。`;
  if (type === "schema" || type === "table") return `数据或 schema 文件，定义 ${p} 相关的数据结构。`;
  if (p.includes("server_args")) return "SGLang 服务参数定义与自动调优逻辑，集中管理启动参数、内存策略和运行时选项。";
  if (p.includes("model_runner")) return "模型执行路径的一部分，负责模型加载、运行、KV cache 初始化或推理执行。";
  if (p.includes("scheduler")) return "调度器相关代码，负责请求调度、批处理、资源管理或运行时状态推进。";
  return `源码文件，属于 ${p.split("/")[0]} 模块，参与 SGLang 的服务、模型执行、测试或工具链实现。`;
}

const nodes = scan.files.map((file) => {
  const type = nodeType(file);
  return {
    id: idFor(file),
    type,
    name: path.basename(file.path),
    filePath: file.path,
    summary: summaryFor(file, type),
    tags: tagsFor(file, type),
    complexity: complexity(file),
  };
});

const nodeIds = new Set(nodes.map((n) => n.id));
const pathToId = new Map(scan.files.map((f) => [f.path, idFor(f)]));
const edges = [];
for (const [sourcePath, targets] of Object.entries(scan.importMap || {})) {
  const source = pathToId.get(sourcePath);
  if (!source) continue;
  for (const targetPath of targets || []) {
    const target = pathToId.get(targetPath);
    if (!target || !nodeIds.has(target)) continue;
    edges.push({ source, target, type: "imports", direction: "outgoing", weight: 0.7 });
  }
}

function layerFor(file) {
  const p = file.path;
  if (p.startsWith("python/sglang/srt/")) return "layer:srt-runtime";
  if (p.startsWith("python/sglang/multimodal_gen/")) return "layer:multimodal-generation";
  if (p.startsWith("python/")) return "layer:python-package";
  if (p.startsWith("sgl-kernel/") || p.startsWith("sgl-router/")) return "layer:native-performance";
  if (p.startsWith("sgl-model-gateway/")) return "layer:model-gateway";
  if (p.startsWith("test/") || p.startsWith("benchmark/")) return "layer:tests-and-benchmarks";
  if (p.startsWith("docs") || p === "README.md") return "layer:documentation";
  if (p.startsWith("docker/") || p.startsWith(".github/") || p.includes("Dockerfile")) return "layer:deployment-and-ci";
  if (p.startsWith("scripts/")) return "layer:tools-and-scripts";
  return "layer:project-root-and-assets";
}

const layerDefs = {
  "layer:srt-runtime": ["SRT Runtime", "核心推理运行时：服务参数、调度器、模型执行器、KV cache、attention 后端和 OpenAI 入口。"],
  "layer:multimodal-generation": ["Multimodal Generation", "多模态生成运行时，包含组件加载、pipeline、平台抽象和多模态推理支持。"],
  "layer:python-package": ["Python Package", "Python 包装层、CLI、benchmark、测试工具和非 SRT 的 Python 支撑代码。"],
  "layer:native-performance": ["Native Performance", "CUDA/C++/Rust/Go 等高性能扩展、kernel、router 和底层构建代码。"],
  "layer:model-gateway": ["Model Gateway", "Rust model gateway 子项目，提供网关、协议和独立服务构建逻辑。"],
  "layer:tests-and-benchmarks": ["Tests and Benchmarks", "回归测试、端到端测试、性能 benchmark 和模型覆盖验证。"],
  "layer:documentation": ["Documentation", "项目 README、用户文档、平台指南和开发说明。"],
  "layer:deployment-and-ci": ["Deployment and CI", "Docker、Kubernetes、GitHub Actions 和部署相关配置。"],
  "layer:tools-and-scripts": ["Tools and Scripts", "维护脚本、wheel 索引更新、构建辅助和仓库自动化工具。"],
  "layer:project-root-and-assets": ["Project Root and Assets", "仓库根配置、资源文件、元数据和未归入其他层的支持文件。"],
};

const grouped = new Map();
for (const file of scan.files) {
  const lid = layerFor(file);
  if (!grouped.has(lid)) grouped.set(lid, []);
  grouped.get(lid).push(idFor(file));
}

const layers = Object.entries(layerDefs).map(([id, [name, description]]) => ({
  id,
  name,
  description,
  nodeIds: grouped.get(id) || [],
}));

function existingNodeId(preferred) {
  for (const p of preferred) {
    const f = scan.files.find((x) => x.path === p);
    if (f) return idFor(f);
  }
  return null;
}

const tourSpecs = [
  ["Project Overview", "从 README 开始理解 SGLang 的定位、支持的模型/硬件和主要能力。", ["README.md"]],
  ["Server Arguments", "阅读服务参数与自动调优入口，理解运行时配置如何影响调度、内存和 CUDA graph。", ["python/sglang/srt/server_args.py"]],
  ["Scheduler Flow", "进入调度器，观察请求如何排队、批处理并推进到模型执行。", ["python/sglang/srt/managers/scheduler.py", "python/sglang/srt/managers/schedule_batch.py"]],
  ["Model Execution", "查看 model runner，理解模型加载、KV cache 初始化和推理执行边界。", ["python/sglang/srt/model_executor/model_runner.py", "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py"]],
  ["Memory And Cache", "学习 memory pool、radix cache 和 KV cache 相关实现。", ["python/sglang/srt/mem_cache/memory_pool.py", "python/sglang/srt/mem_cache/radix_cache.py"]],
  ["Deployment And Tests", "最后看部署配置和测试入口，理解如何验证和运行服务。", ["docker/compose.yaml", "test/README.md"]],
];

const tour = tourSpecs.map(([title, description, paths], i) => {
  const ids = paths.map((p) => existingNodeId([p])).filter(Boolean);
  return { order: i + 1, title, description, nodeIds: ids };
}).filter((s) => s.nodeIds.length > 0);

const graph = {
  version: "1.0.0",
  project: {
    name: scan.name,
    languages: scan.languages,
    frameworks: scan.frameworks,
    description: scan.description,
    analyzedAt: now,
    gitCommitHash: commit,
  },
  nodes,
  edges,
  layers,
  tour,
};

fs.writeFileSync(
  path.join(root, ".understand-anything/intermediate/assembled-graph.json"),
  JSON.stringify(graph, null, 2)
);
fs.writeFileSync(
  path.join(root, ".understand-anything/knowledge-graph.json"),
  JSON.stringify(graph, null, 2)
);
console.log(JSON.stringify({
  nodes: nodes.length,
  edges: edges.length,
  layers: layers.length,
  tour: tour.length,
}, null, 2));
