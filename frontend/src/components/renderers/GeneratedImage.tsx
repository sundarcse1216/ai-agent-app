import { imageUrl } from "../../api/client";

// Deliberately doesn't show the agent's raw text ("✅ Image generated
// successfully.\n\nSaved at:\n{filepath}") — that filepath is a path on
// the server's filesystem, meaningless (and a bit of a leak) to show a
// browser user. A download link off the same image_url is what they
// actually want instead.
export function GeneratedImage({ imageUrl: url }: { imageUrl: string }) {
  const src = imageUrl(url);
  const filename = url.split("/").pop() ?? "image.png";

  return (
    <div className="generated-image">
      <img src={src} alt="AI-generated result" loading="lazy" />
      <a className="download-link" href={src} download={filename}>
        ⬇ Download image
      </a>
    </div>
  );
}
