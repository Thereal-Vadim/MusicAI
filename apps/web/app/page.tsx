import Link from "next/link";

export default function HomePage() {
  return (
    <main>
      <h1>MusicAI</h1>
      <p className="muted">
        Гибридная транскрипция гитары: Demucs + Basic Pitch + MediaPipe + Music Theory Judge.
      </p>
      <div className="card" style={{ marginTop: "1.5rem" }}>
        <p>Создайте draft табулатуры из аудиофайла или YouTube URL.</p>
        <Link className="button" href="/create">
          Создать таб
        </Link>
      </div>
    </main>
  );
}
