# triposr-service

3D reconstruction pipeline: images → GLB via TripoSR → PNG renders via Blender headless.

## Endpoint

`POST /reconstruct`
```json
{ "image_urls": ["https://...", "https://..."] }
```

Returns:
```json
{
  "glb_url": "https://res.cloudinary.com/...",
  "renders": ["https://.../front.png", "https://.../side.png"]
}
```

## Stack
- TripoSR (Stability AI) — CPU inference
- Blender 3.6 headless — Cycles render
- Cloudinary — asset storage
- Railway Pro Plan (8GB+ RAM recommended)

## Env Vars
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_UPLOAD_PRESET`
- `PORT` (default 3001)
