import esbuild from 'esbuild';
import process from 'process';
import builtins from 'builtin-modules';

const isProd = process.argv[2] === 'production';

const ctx = await esbuild.context({
  entryPoints: ['src/main.ts'],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'es2020',
  logLevel: 'info',
  sourcemap: isProd ? false : 'inline',
  treeShaking: true,
  outfile: 'main.js',
  external: ['obsidian', 'electron', 'codemirror', '@codemirror/*', '@lezer/*', ...builtins],
  minify: isProd,
});

if (isProd) {
  await ctx.rebuild();
  await ctx.dispose();
} else {
  await ctx.watch();
}
