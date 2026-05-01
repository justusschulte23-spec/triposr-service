import express from 'express'
import { execFile } from 'child_process'
import { promisify } from 'util'
import fs from 'fs'
import path from 'path'
import https from 'https'
import http from 'http'

const execFileP = promisify(execFile)
const app = express()
app.use(express.json({ limit: '10mb' }))

const TMP = '/tmp/triposr'

// Route download through ScraperAPI if key present, else direct
function proxyUrl(originalUrl) {
  const key = process.env.SCRAPERAPI_KEY
  if (key && key !== 'PLACEHOLDER_ENTER_KEY') {
    return 'https://api.scraperapi.com?api_key=' + key + '&url=' + encodeURIComponent(originalUrl)
  }
  return originalUrl
}

async function downloadFile(url, dest) {
  const fetchUrl = proxyUrl(url)
  return new Promise((resolve, reject) => {
    const proto = fetchUrl.startsWith('https') ? https : http
    const reqOpts = Object.assign(new URL(fetchUrl), {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
      },
    })
    const file = fs.createWriteStream(dest)
    proto.get(reqOpts, res => {
      if (res.statusCode !== 200) {
        file.close(); fs.unlink(dest, () => {})
        reject(new Error('HTTP ' + res.statusCode + ': ' + url))
        return
      }
      res.pipe(file)
      file.on('finish', () => file.close(resolve))
      file.on('error', err => { file.close(); fs.unlink(dest, () => {}); reject(err) })
    }).on('error', err => { file.close(); fs.unlink(dest, () => {}); reject(err) })
  })
}

async function tryDownloadFile(url, dest, label) {
  try { await downloadFile(url, dest); return dest }
  catch (e) { console.warn('[download] skipping ' + label + ': ' + e.message); return null }
}

async function uploadToCloudinary(filePath, folder, resourceType) {
  resourceType = resourceType || 'raw'
  const CLOUD  = process.env.CLOUDINARY_CLOUD_NAME   || 'Poweroflillith'
  const PRESET = process.env.CLOUDINARY_UPLOAD_PRESET || 'poweroflillithvid'
  const { default: FormData } = await import('form-data')
  const { default: fetch }    = await import('node-fetch')
  const form = new FormData()
  form.append('file', fs.createReadStream(filePath))
  form.append('upload_preset', PRESET)
  form.append('folder', folder)
  const resp = await fetch('https://api.cloudinary.com/v1_1/' + CLOUD + '/' + resourceType + '/upload',
    { method: 'POST', body: form })
  const data = await resp.json()
  if (!data.secure_url) throw new Error('Cloudinary upload failed: ' + JSON.stringify(data))
  return data.secure_url
}

app.get('/health', (_, res) => res.json({
  status: 'ok',
  service: 'triposr-service',
  scraper_proxy: !!(process.env.SCRAPERAPI_KEY && process.env.SCRAPERAPI_KEY !== 'PLACEHOLDER_ENTER_KEY'),
}))

app.post('/reconstruct', async (req, res) => {
  res.setTimeout(900000)
  const { image_urls } = req.body || {}
  if (!Array.isArray(image_urls) || image_urls.length === 0)
    return res.status(400).json({ error: 'image_urls array required' })

  const jobId     = Date.now().toString()
  const jobDir    = path.join(TMP, jobId)
  const inputDir  = path.join(jobDir, 'input')
  const outputDir = path.join(jobDir, 'output')
  fs.mkdirSync(inputDir,  { recursive: true })
  fs.mkdirSync(outputDir, { recursive: true })

  try {
    console.log('[' + jobId + '] Downloading ' + Math.min(image_urls.length, 3) + ' images...')
    const results = await Promise.all(
      image_urls.slice(0, 3).map(async (url, i) => {
        const ext  = url.split('?')[0].split('.').pop().replace(/[^a-z0-9]/gi, '') || 'jpg'
        const dest = path.join(inputDir, 'input_' + i + '.' + ext)
        return tryDownloadFile(url, dest, 'image_' + i)
      })
    )
    const imagePaths = results.filter(Boolean)
    if (imagePaths.length === 0)
      return res.status(400).json({ error: 'all_downloads_failed', detail: 'Check SCRAPERAPI_KEY env var' })
    console.log('[' + jobId + '] ' + imagePaths.length + ' images ready')

    const glbPath = path.join(outputDir, 'model.glb')
    console.log('[' + jobId + '] Running TripoSR...')
    const { stdout: tsrOut, stderr: tsrErr } = await execFileP(
      'python3', ['reconstruct.py', imagePaths[0], glbPath],
      { timeout: 600000, maxBuffer: 50 * 1024 * 1024 }
    )
    if (tsrOut) console.log('[tsr] ' + tsrOut.slice(0, 500))
    if (tsrErr) console.error('[tsr] ' + tsrErr.slice(0, 500))
    if (!fs.existsSync(glbPath)) throw new Error('TripoSR did not produce model.glb')

    const frontPng = path.join(outputDir, 'front.png')
    const sidePng  = path.join(outputDir, 'side.png')
    console.log('[' + jobId + '] Blender renders...')
    const { stdout: blOut, stderr: blErr } = await execFileP(
      'blender', ['-b', '-P', 'render.py', '--', glbPath, outputDir],
      { timeout: 300000, maxBuffer: 50 * 1024 * 1024 }
    )
    if (blOut) console.log('[blender] ' + blOut.slice(0, 500))
    if (blErr) console.error('[blender] ' + blErr.slice(0, 500))

    console.log('[' + jobId + '] Uploading...')
    const [glbUrl, frontUrl, sideUrl] = await Promise.all([
      uploadToCloudinary(glbPath, '3d_models', 'raw'),
      fs.existsSync(frontPng) ? uploadToCloudinary(frontPng, '3d_renders', 'image') : Promise.resolve(null),
      fs.existsSync(sidePng)  ? uploadToCloudinary(sidePng,  '3d_renders', 'image') : Promise.resolve(null),
    ])
    res.json({ glb_url: glbUrl, renders: [frontUrl, sideUrl].filter(Boolean), job_id: jobId })

  } catch (err) {
    console.error('[' + jobId + '] Error:', err.message)
    if (!res.headersSent) res.status(500).json({ error: 'reconstruction_failed', detail: err.message })
  } finally {
    try { fs.rmSync(jobDir, { recursive: true, force: true }) } catch {}
  }
})

const port = process.env.PORT || 3001
const server = app.listen(port, () => console.log('triposr-service on port ' + port))
server.requestTimeout = 900000
server.headersTimeout  = 910000
