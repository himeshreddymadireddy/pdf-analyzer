import { z } from "zod";
import { publicProcedure, router } from "../_core/trpc";
import { processDocumentBuffer, summarizeDocument, answerQuestion } from "../documentService";

export const analyzerRouter = router({
  processDocument: publicProcedure
    .input(
      z.object({
        fileName: z.string(),
        fileBase64: z.string(),
      })
    )
    .mutation(async ({ input }) => {
      const buffer = Buffer.from(input.fileBase64, "base64");
      const result = await processDocumentBuffer(buffer, input.fileName);
      if (!result.success) {
        throw new Error(result.error || "Failed to process document.");
      }
      return result;
    }),

  summarize: publicProcedure
    .input(
      z.object({
        pages: z.array(z.any()),
        docType: z.string(),
        provider: z.string(),
        model: z.string(),
        apiKey: z.string().optional(),
      })
    )
    .mutation(async ({ input }) => {
      const summary = await summarizeDocument(
        input.pages,
        input.docType,
        input.provider,
        input.model,
        input.apiKey
      );
      return { summary };
    }),

  askQuestion: publicProcedure
    .input(
      z.object({
        pages: z.array(z.any()),
        docType: z.string(),
        question: z.string(),
        provider: z.string(),
        model: z.string(),
        apiKey: z.string().optional(),
      })
    )
    .mutation(async ({ input }) => {
      const result = await answerQuestion(
        input.pages,
        input.docType,
        input.question,
        input.provider,
        input.model,
        input.apiKey
      );
      return result;
    }),
});
