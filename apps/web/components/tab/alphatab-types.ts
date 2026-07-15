/** Narrow AlphaTab runtime types used by the viewer / track selector. */

export type AlphaTabScoreTrack = {
  index: number;
  name?: string;
};

export type AlphaTabScore = {
  tracks: AlphaTabScoreTrack[];
};

export type AlphaTabApi = {
  destroy: () => void;
  load: (data: unknown, trackIndexes?: number[]) => boolean;
  tex: (tex: string) => void;
  playPause: () => void;
  stop: () => void;
  playbackSpeed: number;
  playerStateChanged: { on: (cb: (args: { state: number }) => void) => void };
  scoreLoaded: { on: (cb: (score?: AlphaTabScore) => void) => void };
  renderStarted: { on: (cb: () => void) => void };
  error: { on: (cb: (err: { message: string }) => void) => void };
  timePosition: number;
  endTime: number;
  score?: AlphaTabScore | null;
  renderTracks: (tracks: AlphaTabScoreTrack[]) => void;
  changeTrackMute: (tracks: AlphaTabScoreTrack[], mute: boolean) => void;
  changeTrackSolo: (tracks: AlphaTabScoreTrack[], solo: boolean) => void;
  changeTrackVolume: (tracks: AlphaTabScoreTrack[], volume: number) => void;
};
