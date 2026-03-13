// Structured JSON logger — no external deps
const logger = {
  info(data: Record<string, unknown>, msg: string) {
    console.log(JSON.stringify({ level: "info", msg, ...data, ts: new Date().toISOString() }));
  },
  error(data: Record<string, unknown>, msg: string) {
    console.error(JSON.stringify({ level: "error", msg, ...data, ts: new Date().toISOString() }));
  },
  warn(data: Record<string, unknown>, msg: string) {
    console.warn(JSON.stringify({ level: "warn", msg, ...data, ts: new Date().toISOString() }));
  },
};
export { logger };
