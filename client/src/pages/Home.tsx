import React, { useState, useEffect, useRef, useMemo } from "react";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Loader2, Upload, FileText, Presentation, Key, RefreshCw, Send, CheckCircle2, AlertCircle, ShieldCheck } from "lucide-react";
import { Streamdown } from "streamdown";
import { toast } from "sonner";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [provider, setProvider] = useState<string>("Anthropic");
  const [model, setModel] = useState<string>("claude-sonnet-4-6");
  const [apiKey, setApiKey] = useState<string>("");
  const [enableFallback, setEnableFallback] = useState<boolean>(false);
  const [fallbackProvider, setFallbackProvider] = useState<string>("OpenAI");
  const [fallbackModel, setFallbackModel] = useState<string>("gpt-4o");
  const [fallbackApiKey, setFallbackApiKey] = useState<string>("");

  const [activeTab, setActiveTab] = useState<string>("summary");
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [progressMsg, setProgressMsg] = useState<string>("");

  // Summary & Q&A states
  const [summaryData, setSummaryData] = useState<string>("");
  const [isSummarizing, setIsSummarizing] = useState<boolean>(false);
  const [question, setQuestion] = useState<string>("");
  const [qaHistory, setQaHistory] = useState<Array<{ q: string; a: string; citations?: any[] }>>([]);
  const [isAnswering, setIsAnswering] = useState<boolean>(false);

  // Settings fingerprint for automatic cache invalidation
  const settingsFingerprint = useMemo(() => {
    return `${provider}:${model}:${apiKey}:${enableFallback}:${fallbackProvider}:${fallbackModel}:${fallbackApiKey}`;
  }, [provider, model, apiKey, enableFallback, fallbackProvider, fallbackModel, fallbackApiKey]);

  const prevFingerprintRef = useRef(settingsFingerprint);

  useEffect(() => {
    if (prevFingerprintRef.current !== settingsFingerprint) {
      prevFingerprintRef.current = settingsFingerprint;
      if (analysisResult) {
        setSummaryData("");
        setQaHistory([]);
        toast.info("Settings or API key changed. Cleared cached document outputs for fresh generation.");
      }
    }
  }, [settingsFingerprint, analysisResult]);

  const processMutation = trpc.analyzer.processDocument.useMutation();
  const summarizeMutation = trpc.analyzer.summarize.useMutation();
  const qaMutation = trpc.analyzer.askQuestion.useMutation();

  const handleFileUpload = async (uploadedFile: File) => {
    if (uploadedFile.size > 50 * 1024 * 1024) {
      toast.error("File size exceeds 50 MB limit.");
      return;
    }
    setFile(uploadedFile);
    setIsProcessing(true);
    setProgressPercent(20);
    setProgressMsg("Reading document stream & uploading...");

    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const base64String = (reader.result as string).split(",")[1];
        setProgressPercent(50);
        setProgressMsg("Running Python extraction & layout structuring...");
        
        const result = await processMutation.mutateAsync({
          fileName: uploadedFile.name,
          fileBase64: base64String,
        });

        setAnalysisResult(result);
        setProgressPercent(80);
        setProgressMsg("Generating executive summary via LLM...");

        const summaryRes = await summarizeMutation.mutateAsync({
          pages: result.type === "pdf" ? result.pages : result.slides,
          docType: result.type,
          provider,
          model,
          apiKey: apiKey || undefined,
        });

        setSummaryData(summaryRes.summary);
        setProgressPercent(100);
        setIsProcessing(false);
        setProgressMsg("");
        toast.success(`Successfully processed and summarized ${result.display_name}`);
      } catch (err: any) {
        setIsProcessing(false);
        setProgressMsg("");
        toast.error(`Processing error: ${err.message}`);
      }
    };
    reader.onerror = () => {
      setIsProcessing(false);
      toast.error("Failed to read file.");
    };
    reader.readAsDataURL(uploadedFile);
  };

  const handleRegenerateSummary = async () => {
    if (!analysisResult) return;
    setIsSummarizing(true);
    try {
      const summaryRes = await summarizeMutation.mutateAsync({
        pages: analysisResult.type === "pdf" ? analysisResult.pages : analysisResult.slides,
        docType: analysisResult.type,
        provider,
        model,
        apiKey: apiKey || undefined,
      });
      setSummaryData(summaryRes.summary);
      toast.success("Summary regenerated successfully with current provider settings.");
    } catch (err: any) {
      toast.error(`Regeneration failed: ${err.message}`);
    } finally {
      setIsSummarizing(false);
    }
  };

  const handleAskQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || !analysisResult) return;
    const q = question;
    setQuestion("");
    setIsAnswering(true);

    try {
      const res = await qaMutation.mutateAsync({
        pages: analysisResult.type === "pdf" ? analysisResult.pages : analysisResult.slides,
        docType: analysisResult.type,
        question: q,
        provider,
        model,
        apiKey: apiKey || undefined,
      });

      setQaHistory((prev) => [...prev, { q, a: res.answer, citations: res.citations }]);
    } catch (err: any) {
      toast.error(`Q&A error: ${err.message}`);
    } finally {
      setIsAnswering(false);
    }
  };

  return (
    <div className="min-h-screen cinematic-glow text-foreground flex flex-col">
      {/* Header */}
      <header className="border-b border-border/40 px-6 py-4 flex items-center justify-between glass-panel sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400 font-bold shadow-[0_0_15px_rgba(212,175,55,0.2)]">
            ⚡
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-wider golden-gradient-text uppercase">PDF & PPTX Intelligence</h1>
            <p className="text-xs text-muted-foreground">Cinematic Document Analysis & Grounded Q&A</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="border-amber-500/30 text-amber-400 text-xs">
            Chiaroscuro Engine active
          </Badge>
        </div>
      </header>

      {/* Main Content Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Configuration & Upload */}
        <div className="lg:col-span-4 space-y-6">
          {/* Provider & Credentials Card */}
          <Card className="glass-panel border-border/40">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold uppercase tracking-wider text-amber-400">
                1. Provider & Model Configuration
              </CardTitle>
              <CardDescription className="text-xs">
                Configure primary & fallback LLM parameters and API keys.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-xs">
              <div>
                <Label className="text-muted-foreground">Primary Provider</Label>
                <select
                  value={provider}
                  onChange={(e) => {
                    setProvider(e.target.value);
                    if (e.target.value === "Anthropic") setModel("claude-sonnet-4-6");
                    else if (e.target.value === "OpenAI") setModel("gpt-4o");
                    else setModel("deepseek-chat");
                  }}
                  className="w-full mt-1 bg-background/80 border border-border rounded p-2 text-foreground"
                >
                  <option value="Anthropic">Anthropic (Claude)</option>
                  <option value="OpenAI">OpenAI (GPT)</option>
                  <option value="DeepSeek">DeepSeek</option>
                </select>
              </div>

              <div>
                <Label className="text-muted-foreground">Model</Label>
                <Input
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="mt-1 bg-background/80 border-border text-xs"
                />
              </div>

              <div>
                <div className="flex justify-between items-center">
                  <Label className="text-muted-foreground">API Key Override (Optional)</Label>
                  <span className={`text-[10px] flex items-center gap-1 ${apiKey ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {apiKey ? <ShieldCheck className="w-3 h-3" /> : <AlertCircle className="w-3 h-3" />}
                    {apiKey ? "Custom Key Active" : "Using System Default"}
                  </span>
                </div>
                <div className="relative mt-1">
                  <Input
                    type="password"
                    placeholder="sk-..."
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    className="bg-background/80 border-border text-xs pr-8"
                  />
                  <Key className="w-3.5 h-3.5 text-amber-400 absolute right-2.5 top-2.5" />
                </div>
                <p className="text-[10px] text-muted-foreground mt-1">
                  Changing keys automatically invalidates caches for fresh output.
                </p>
              </div>

              {/* Fallback Toggle */}
              <div className="pt-2 border-t border-border/40 flex items-center justify-between">
                <div>
                  <Label className="font-medium">Enable Fallback Model</Label>
                  <p className="text-[10px] text-muted-foreground">Auto-retry with secondary provider</p>
                </div>
                <Switch checked={enableFallback} onCheckedChange={setEnableFallback} />
              </div>

              {enableFallback && (
                <div className="space-y-3 pt-2 pl-2 border-l border-amber-500/30">
                  <div>
                    <Label className="text-muted-foreground">Fallback Provider</Label>
                    <select
                      value={fallbackProvider}
                      onChange={(e) => setFallbackProvider(e.target.value)}
                      className="w-full mt-1 bg-background/80 border border-border rounded p-2 text-foreground text-xs"
                    >
                      <option value="OpenAI">OpenAI</option>
                      <option value="Anthropic">Anthropic</option>
                      <option value="DeepSeek">DeepSeek</option>
                    </select>
                  </div>
                  <div>
                    <Label className="text-muted-foreground">Fallback API Key</Label>
                    <Input
                      type="password"
                      placeholder="sk-..."
                      value={fallbackApiKey}
                      onChange={(e) => setFallbackApiKey(e.target.value)}
                      className="bg-background/80 border-border text-xs"
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Upload Card */}
          <Card className="glass-panel border-border/40">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold uppercase tracking-wider text-amber-400">
                2. Document Upload
              </CardTitle>
              <CardDescription className="text-xs">
                PDF or PPTX up to 50 MB with drag-and-drop support.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (e.dataTransfer.files?.[0]) {
                    handleFileUpload(e.dataTransfer.files[0]);
                  }
                }}
                className="border-2 border-dashed border-border/60 hover:border-amber-500/50 rounded-lg p-6 text-center cursor-pointer transition-colors bg-background/40"
              >
                <input
                  type="file"
                  accept=".pdf,.pptx"
                  className="hidden"
                  id="file-input"
                  onChange={(e) => {
                    if (e.target.files?.[0]) handleFileUpload(e.target.files[0]);
                  }}
                />
                <label htmlFor="file-input" className="cursor-pointer space-y-2 block">
                  <Upload className="w-8 h-8 text-amber-400 mx-auto" />
                  <p className="text-xs font-medium">Click to upload or drag & drop</p>
                  <p className="text-[10px] text-muted-foreground">PDF or PPTX (Max 50MB)</p>
                </label>
              </div>

              {file && (
                <div className="mt-4 p-3 bg-secondary/50 rounded flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2 truncate">
                    {file.name.endsWith(".pdf") ? (
                      <FileText className="w-4 h-4 text-amber-400 shrink-0" />
                    ) : (
                      <Presentation className="w-4 h-4 text-amber-400 shrink-0" />
                    )}
                    <span className="truncate">{file.name}</span>
                  </div>
                  <Badge variant="outline" className="text-[10px]">
                    {(file.size / (1024 * 1024)).toFixed(1)} MB
                  </Badge>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Analysis & Interactive Workspace */}
        <div className="lg:col-span-8 space-y-6">
          {isProcessing ? (
            <Card className="glass-panel border-border/40 p-12 text-center space-y-6">
              <Loader2 className="w-10 h-10 text-amber-400 animate-spin mx-auto" />
              <div className="space-y-2 max-w-md mx-auto">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Processing Document Intelligence</span>
                  <span>{progressPercent}%</span>
                </div>
                <Progress value={progressPercent} className="h-2 bg-secondary" />
                <p className="text-xs text-amber-400 font-mono">{progressMsg}</p>
              </div>
            </Card>
          ) : !analysisResult ? (
            <Card className="glass-panel border-border/40 p-12 text-center space-y-4">
              <div className="w-12 h-12 rounded-full bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mx-auto text-amber-400 shadow-[0_0_20px_rgba(212,175,55,0.15)]">
                📄
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-semibold">No Document Loaded</h3>
                <p className="text-xs text-muted-foreground max-w-sm mx-auto">
                  Upload a PDF or PPTX document from the left panel to begin map-reduce summarization and grounded Q&A.
                </p>
              </div>
            </Card>
          ) : (
            <Card className="glass-panel border-border/40">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <div className="px-6 pt-4 border-b border-border/40 flex items-center justify-between flex-wrap gap-4">
                  <div>
                    <h2 className="text-sm font-bold">{analysisResult.display_name}</h2>
                    <p className="text-[10px] text-muted-foreground">
                      {analysisResult.type === "pdf"
                        ? `${analysisResult.pages.length} Pages Extracted`
                        : `${analysisResult.slides.length} Slides Extracted`}
                    </p>
                  </div>
                  <TabsList className="bg-background/60 border border-border/50">
                    <TabsTrigger value="summary" className="text-xs">Summary & Insights</TabsTrigger>
                    <TabsTrigger value="qa" className="text-xs">Grounded Q&A</TabsTrigger>
                    <TabsTrigger value="raw" className="text-xs">Extracted Content</TabsTrigger>
                  </TabsList>
                </div>

                <div className="p-6">
                  <TabsContent value="summary" className="space-y-4 m-0">
                    <div className="flex items-center justify-between">
                      <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-400">
                        Executive Summary & Map-Reduce Analysis
                      </h3>
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={isSummarizing}
                        className="h-7 text-xs border-amber-500/30 text-amber-400 hover:bg-amber-500/10"
                        onClick={handleRegenerateSummary}
                      >
                        {isSummarizing ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <RefreshCw className="w-3 h-3 mr-1" />}
                        Regenerate
                      </Button>
                    </div>
                    <div className="bg-background/50 border border-border/40 rounded-lg p-4 text-xs leading-relaxed max-h-[500px] overflow-y-auto">
                      <Streamdown>{summaryData}</Streamdown>
                    </div>
                  </TabsContent>

                  <TabsContent value="qa" className="space-y-4 m-0">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-400">
                      {analysisResult.type === "pdf" ? "BM25 Grounded Q&A" : "Full-Deck Presentation Q&A"}
                    </h3>

                    <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
                      {qaHistory.length === 0 ? (
                        <p className="text-xs text-muted-foreground text-center py-8">
                          Ask a question about the document to receive cited, grounded answers.
                        </p>
                      ) : (
                        qaHistory.map((item, idx) => (
                          <div key={idx} className="space-y-2 bg-background/40 border border-border/40 rounded-lg p-3 text-xs">
                            <p className="font-semibold text-amber-400">Q: {item.q}</p>
                            <div className="text-foreground">
                              <Streamdown>{item.a}</Streamdown>
                            </div>
                            {item.citations && item.citations.length > 0 && (
                              <div className="flex items-center gap-1 pt-1 text-[10px] text-amber-400">
                                <span>Citations:</span>
                                {item.citations.map((c: any, cIdx: number) => (
                                  <Badge key={cIdx} variant="outline" className="text-[9px] border-amber-500/30 text-amber-300">
                                    {c.page ? `Page ${c.page}` : c.slide ? `Slide ${c.slide}` : JSON.stringify(c)}
                                  </Badge>
                                ))}
                              </div>
                            )}
                          </div>
                        ))
                      )}
                      {isAnswering && (
                        <div className="flex items-center gap-2 text-xs text-muted-foreground py-2">
                          <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-400" />
                          Retrieving evidence & generating grounded response...
                        </div>
                      )}
                    </div>

                    <form onSubmit={handleAskQuestion} className="flex gap-2 pt-2">
                      <Input
                        placeholder={
                          analysisResult.type === "pdf"
                            ? "Ask a question about the PDF (BM25 search)..."
                            : "Ask a question about the presentation..."
                        }
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        className="bg-background/80 border-border text-xs flex-1"
                      />
                      <Button type="submit" size="sm" disabled={isAnswering} className="bg-amber-500 text-black hover:bg-amber-400 text-xs">
                        {isAnswering ? <Loader2 className="w-3 h-3 animate-spin" /> : <Send className="w-3 h-3" />}
                      </Button>
                    </form>
                  </TabsContent>

                  <TabsContent value="raw" className="space-y-4 m-0">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-400">
                      Extracted Text Layers
                    </h3>
                    <div className="space-y-3 max-h-96 overflow-y-auto pr-2 text-xs">
                      {analysisResult.type === "pdf" ? (
                        analysisResult.pages.map((p: any) => (
                          <div key={p.page_number} className="bg-background/40 border border-border/40 rounded p-3 space-y-1">
                            <div className="flex justify-between text-[10px] text-muted-foreground">
                              <span>Page {p.page_number}</span>
                              <span>Method: {p.method}</span>
                            </div>
                            <p className="font-mono text-[11px] whitespace-pre-wrap">{p.text}</p>
                          </div>
                        ))
                      ) : (
                        analysisResult.slides.map((s: string, i: number) => (
                          <div key={i} className="bg-background/40 border border-border/40 rounded p-3 space-y-1">
                            <span className="text-[10px] text-muted-foreground">Slide {i + 1}</span>
                            <p className="font-mono text-[11px] whitespace-pre-wrap">{s}</p>
                          </div>
                        ))
                      )}
                    </div>
                  </TabsContent>
                </div>
              </Tabs>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}
