import { describe, expect, it } from "vitest";
import { processDocumentBuffer } from "./documentService";

describe("Document Service Bridge", () => {
  it("handles empty or invalid buffer gracefully", async () => {
    const emptyBuffer = Buffer.from("");
    const result = await processDocumentBuffer(emptyBuffer, "test.pdf");
    expect(result.success).toBe(false);
  });
});
