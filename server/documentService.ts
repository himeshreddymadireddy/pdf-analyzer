import { spawn } from "child_process";
import path from "path";

export interface ExtractedPage {
  page_number: number;
  text: string;
  method: string;
  warnings: string[];
}

export interface ExtractedPdfResult {
  success: boolean;
  type: "pdf";
  display_name: string;
  document_id: string;
  ocr_status: string;
  pages: ExtractedPage[];
  warnings: string[];
  error?: string;
}

export interface ExtractedPptxResult {
  success: boolean;
  type: "pptx";
  display_name: string;
  slides: string[];
  error?: string;
}

export async function processDocumentBuffer(fileBuffer: Buffer, fileName: string): Promise<ExtractedPdfResult | ExtractedPptxResult> {
  return new Promise((resolve, reject) => {
    const pythonScript = path.resolve(__dirname, "python/bridge.py");
    const py = spawn("python3", [pythonScript]);

    let stdoutData = "";
    let stderrData = "";

    py.stdout.on("data", (data) => {
      stdoutData += data.toString();
    });

    py.stderr.on("data", (data) => {
      stderrData += data.toString();
    });

    py.on("close", (code) => {
      if (code !== 0) {
        return resolve({
          success: false,
          type: fileName.toLowerCase().endsWith(".pdf") ? "pdf" : "pptx",
          display_name: fileName,
          document_id: "",
          ocr_status: "error",
          pages: [],
          slides: [],
          warnings: [],
          error: `Python process exited with code ${code}: ${stderrData}`
        } as any);
      }
      try {
        const result = JSON.parse(stdoutData);
        resolve(result);
      } catch (err) {
        resolve({
          success: false,
          type: fileName.toLowerCase().endsWith(".pdf") ? "pdf" : "pptx",
          display_name: fileName,
          document_id: "",
          ocr_status: "error",
          pages: [],
          slides: [],
          warnings: [],
          error: `Failed to parse Python bridge output: ${stdoutData} | stderr: ${stderrData}`
        } as any);
      }
    });

    const payload = JSON.stringify({
      action: "process",
      file_name: fileName,
      file_data: fileBuffer.toString("base64")
    });

    py.stdin.write(payload);
    py.stdin.end();
  });
}

export async function summarizeDocument(pages: any[], docType: string, provider: string, model: string, apiKey?: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const pythonScript = path.resolve(__dirname, "python/bridge.py");
    const py = spawn("python3", [pythonScript]);

    let stdoutData = "";
    let stderrData = "";

    py.stdout.on("data", (data) => {
      stdoutData += data.toString();
    });

    py.stderr.on("data", (data) => {
      stderrData += data.toString();
    });

    py.on("close", (code) => {
      if (code !== 0) {
        return reject(new Error(`Python summarize process failed: ${stderrData}`));
      }
      try {
        const res = JSON.parse(stdoutData);
        if (!res.success) {
          return reject(new Error(res.error || "Summarization failed"));
        }
        resolve(res.summary);
      } catch (err) {
        reject(new Error(`Failed to parse summary output: ${stdoutData}`));
      }
    });

    const payload = JSON.stringify({
      action: "summarize",
      pages,
      doc_type: docType,
      provider,
      model,
      api_key: apiKey
    });

    py.stdin.write(payload);
    py.stdin.end();
  });
}

export async function answerQuestion(pages: any[], docType: string, question: string, provider: string, model: string, apiKey?: string): Promise<{ answer: string; citations: any[] }> {
  return new Promise((resolve, reject) => {
    const pythonScript = path.resolve(__dirname, "python/bridge.py");
    const py = spawn("python3", [pythonScript]);

    let stdoutData = "";
    let stderrData = "";

    py.stdout.on("data", (data) => {
      stdoutData += data.toString();
    });

    py.stderr.on("data", (data) => {
      stderrData += data.toString();
    });

    py.on("close", (code) => {
      if (code !== 0) {
        return reject(new Error(`Python QA process failed: ${stderrData}`));
      }
      try {
        const res = JSON.parse(stdoutData);
        if (!res.success) {
          return reject(new Error(res.error || "QA failed"));
        }
        resolve({ answer: res.answer, citations: res.citations });
      } catch (err) {
        reject(new Error(`Failed to parse QA output: ${stdoutData}`));
      }
    });

    const payload = JSON.stringify({
      action: "qa",
      pages,
      doc_type: docType,
      question,
      provider,
      model,
      api_key: apiKey
    });

    py.stdin.write(payload);
    py.stdin.end();
  });
}
