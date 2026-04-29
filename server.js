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

// ── Helpers ──────────────────────────────────────────────────────────────────

async function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const proto = url.startsWith('https') ? https : http
    const file = fs.createWriteStream(dest)
    proto.get(url, res => {
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode}: ${url}`))
        return
      }
      res.pipe(file)
      file.on('finish', () => file.close(resolve))
      file.on('error', reject)
    }).on('error', reject)
  })
}

async function uploadToCloudinary(filePath, folder, resourceType = 'raw') {
  const CLOUD = process.env.CLOUDINARY_CLOUD_NAME || 'Poweroflillith'
  const PRESET = process.env.CLOUDINARY_UPLOAD_PRESET || 'poweroflillithvid'

  const { default: FormData } = await import('form-data')
  const { default: fetch } = await import('node-fetch')

  const form = new FormData()
  form.append('file', fs.createReadStream(filePath))
  form.append('upload_preset', PRESET)
  form.append('folder', folder)

  const resp = await fetch(
    `https://api.cloudinary.com/v1_1/${CLOUD}/${resourceType}/upload`,
    { method: 'POST', body: form }
  )
  const data = await resp.json()
  if (!data.secure_url) throw new Error(`Cloudinary upload failed: ${JSON.stringify(data)}`)
  return data.secure_url
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get('/health', (_, res) => res.json({ status: 'ok', service: 'triposr-service' }))

app.post('/reconstruct', async (req, res) => {
  res.setTimeout(900000) // 15 min timeout

  const { image_urls } = req.body || {}
  if (!Array.isArray(image_urls) || image_urls.length === 0) {
    return res.status(400).json({ error: 'image_urls array required' })
  }

  const jobId = Date.now().toString()
  const jobDir = path.join(TMP, jobId)
  const inputDir = path.join(jobDir, 'input')
  const outputDir = path.join(jobDir, 'output')

  fs.mkdirSync(inputDir, { recursive: true })
  fs.mkdirSync(outputDir, { recursive: true })

  try {
    // 1) Download all input images
    console.log(`[${jobId}] Downloading ${image_urls.length} images...`)
    const imagePaths = await Promise.all(
      image_urls.slice(0, 3).map(async (url, i) => {
        const ext = url.split('?')[0].split('.').pop() || 'jpg'
        const dest = path.join(inputDir, `input_${i}.${ext}`)
        await downloadFile(url, dest)
        return dest
      })
    )
    console.log(`[${jobId}] Images ready: ${imagePaths.length}`)

    // 2) Run TripoSR Python script (uses first image as primary)
    console.log(`[${jobId}] Running TripoSR reconstruction...`)
    const glbPath = path.join(outputDir, 'model.glb')

    const { stdout: tsrOut, stderr: tsrErr } = await execFileP(
      'python3',
      ['reconstruct.py', imagePaths[0], glbPath],
      { timeout: 600000, maxBuffer: 50 * 1024 * 1024 }
    )
    if (tsrOut) console.log(`[tsr] ${tsrOut.slice(0, 500)}`)
    if (tsrErr) console.error(`[tsr stderr] ${tsrErr.slice(0, 500)}`)

    if (!fs.existsSync(glbPath)) {
      throw new Error('TripoSR did not produce model.glb')
    }
    console.log(`[${jobId}] GLB created: ${fs.statSync(glbPath).size} bytes`)

    // 3) Blender headless: render front + side view
    console.log(`[${jobId}] Running Blender renders...`)
    const frontPng = path.join(outputDir, 'front.png')
    const sidePng = path.join(outputDir, 'side.png')

    const { stdout: blOut, stderr: blErr } = await execFileP(
      'blender',
      ['-b', '-P', 'render.py', '--', glbPath, outputDir],
      { timeout: 300000, maxBuffer: 50 * 1024 * 1024 }
    )
    if (blOut) console.log(`[blender] ${blOut.slice(0, 500)}`)
    if (blErr) console.error(`[blender stderr] ${blErr.slice(0, 500)}`)

    // 4) Upload to Cloudinary
    console.log(`[${jobId}] Uploading to Cloudinary...`)
    const [glbUrl, frontUrl, sideUrl] = await Promise.all([
      uploadToCloudinary(glbPath, '3d_models', 'raw'),
      fs.existsSync(frontPng) ? uploadToCloudinary(frontPng, '3d_renders', 'image') : Promise.resolve(null),
      fs.existsSync(sidePng)  ? uploadToCloudinary(sidePng,  '3d_renders', 'image') : Promise.resolve(null),
    ])

    console.log(`[${jobId}] Done.`)
    res.json({
      glb_url: glbUrl,
      renders: [frontUrl, sideUrl].filter(Boolean),
      job_id: jobId,
    })

  } catch (err) {
    console.error(`[${jobId}] Error:`, err.message)
    if (!res.headersSent) {
      res.status(500).json({ error: 'reconstruction_failed', detail: err.message })
    }
  } finally {
    // Cleanup
    try { fs.rmSync(jobDir, { recursive: true, force: true }) } catch {}
  }
})

const port = process.env.PORT || 3001
const server = app.listen(port, () => console.log(`triposr-service on port ${port}`))
server.requestTimeout = 900000
server.headersTimeout  = 910000
