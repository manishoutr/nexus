import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build is written into app/static so FastAPI serves it with no extra server.
// emptyOutDir is false because eval.html and eval/ live there too.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../app/static", emptyOutDir: false },
  server: { proxy: { "/api": "http://localhost:8000" } },
  test: { environment: "jsdom", setupFiles: "./src/setup.js", globals: true },
});
