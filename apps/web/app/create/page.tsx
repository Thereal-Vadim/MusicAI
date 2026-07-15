import { UploadForm } from "@/components/upload-form/UploadForm";

export default function CreatePage() {
  return (
    <main>
      <h1>Создать таб</h1>
      <p className="muted">Upload или YouTube URL → AI pipeline → draft с подсветкой спорных мест.</p>
      <div style={{ marginTop: "1.5rem" }}>
        <UploadForm />
      </div>
    </main>
  );
}
