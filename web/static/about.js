// About page: what the project is, how to run both apps, how the browser version is built and checked, credits.
import { route, app, api, esc, isStatic } from './core.js';

route('about', async () => {
  const [{ demos, note }, { operations }] = await Promise.all([api('/demos'), api('/ops')]);
  app().innerHTML = `
  <h1>About</h1>
  <p class="lead">The Advanced TIFF Stack Viewer is a PyQt6 / pyqtgraph desktop application for viewing, analysing and
  processing microscopy TIFF stacks, designed for fluorescence stacks overlaid with a binary mask for region-specific
  analysis. This site is a browser companion built from the same repository.</p>

  <h2>Run the desktop app</h2>
  <pre>pip install -r requirements.txt
python main.py                                  # or
python main.py -f fluorescence_stack.tif -m mask_stack.tif</pre>
  <p>Desktop features (README): fluorescence + mask overlay with opacity and colour, frame navigation, histogram and
  contrast, Z-projections, ROI tools, time series, statistics, PCA and clustering, registration, filters and export.</p>

  <h2>Run the web app</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py          # http://127.0.0.1:8128 (local only)</pre>
  <p>${isStatic() ? 'You are on the static site (GitHub Pages).' : 'You are on the local app.'} Both read TIFFs, draw,
  process and analyse in the browser; the local app adds the operations that exist only in Python
  (${operations.filter(o => o.where === 'python').map(o => esc(o.label)).join('; ')}), and can run every ported operation in Python too.</p>

  <h2>How the browser version works</h2>
  <ul>
    <li><b>TIFF reader</b> (<code>tiff_reader.js</code>, no library): classic TIFF and BigTIFF, either byte order, 8/16/32-bit
      integer and 32/64-bit float, strips or tiles, uncompressed, LZW, Deflate or PackBits with horizontal or
      floating-point predictor, one or more samples per pixel. Stacks are laid out from ImageJ hyperstack descriptions,
      OME-XML or tifffile's shaped descriptions, otherwise every page is a frame; time points × z-slices become frames.
      JPEG/ZSTD-compressed files are refused with a message. Tested on a set of files written by tifffile
      (<code>scripts/make_tiff_fixtures.py</code>): shapes, types, scales and pixel checksums match.</li>
    <li><b>Pixel size</b>: from the ImageJ unit and XResolution, OME PhysicalSizeX, or a resolution in centimetres; the viewer
      states which one it used, and shows pixels when the file has none. Read-out coordinates are 0-based pixel indices
      from the top-left corner (column x, row y).</li>
    <li><b>Display</b>: each channel goes through its LUT between min and max (Auto = 1st–99th percentile, as the desktop's
      auto levels) and channels are added; the mask is mixed in with the desktop formula
      <span class="eq">out = base·(1 − α·m) + colour·α·m</span> (<code>core/overlay_blend.py</code>). The desktop shows one
      fluorescence stack and a mask; several channels at once is this viewer's addition (a ROADMAP item).
      LUTs: the desktop colormaps via <code>utils.helpers.convert_colormap_to_lut</code>, plus single-colour ramps in the mask colours.</li>
    <li><b>Processing and analysis</b>: JavaScript ports of <code>ImageStack</code>, <code>ImageProcessor</code> and
      <code>DataAnalyzer</code> functions. <code>scripts/make_ops_fixture.py</code> runs the Python code on a crop of
      the demo, and <code>ops.test.mjs</code> requires the ports to agree (integer filter outputs may differ by one
      count where the float result sits on an integer boundary). Heavy work runs in a Web Worker. Results become new
      channels or the mask; the original data are never overwritten.</li>
    <li>The "bilateral" filter of the repository runs a Gaussian of <code>sigma_spatial</code> (it is not a true bilateral
      filter); the browser does the same and labels it so.</li>
  </ul>

  <h2>Demo data</h2>
  <p class="callout">${esc(note)}</p>
  <ul>${demos.map(d => `<li><b>${esc(d.file.split('/').pop())}</b>: ${esc(d.text)}</li>`).join('')}</ul>

  <h2>Credits</h2>
  <p>Advanced TIFF Stack Viewer by George Dickinson, MIT License. Built with PyQt6, pyqtgraph, NumPy, SciPy,
  scikit-image, tifffile, OpenCV, pandas, scikit-learn and matplotlib; the web interface uses FastAPI and plain
  JavaScript modules. <a href="https://github.com/gddickinson/overlay">Source on GitHub</a>.</p>`;
});
