const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function client() {
  const html = fs.readFileSync(path.join(__dirname, '../examples/transports/voice-client.html'), 'utf8');
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  function element() {
    return {
      children: [], textContent: '', hidden: false, scrollTop: 0, scrollHeight: 0, clientHeight: 0,
      append(...nodes) { this.children.push(...nodes); },
      replaceChildren() { this.children = []; },
      set innerHTML(value) { throw new Error('Conversation content must be rendered as text'); },
    };
  }
  const elements = new Map();
  const context = vm.createContext({
    document: {
      querySelector(selector) {
        if (!elements.has(selector)) elements.set(selector, element());
        return elements.get(selector);
      },
      createElement: element,
      body: { dataset: {}, append() {} },
    },
    window: { addEventListener() {} },
    Uint8Array, DataView, ArrayBuffer, Set, Math,
  });
  vm.runInContext(script, context);
  const nodes = [];
  const state = {
    playing: new Set(), messages: new Map(), nextTime: 0, tail: new Uint8Array(0),
    context: {
      currentTime: 1,
      destination: {},
      createBuffer(channels, length, rate) {
        const data = new Float32Array(length);
        return { duration: length / rate, data, getChannelData: () => data };
      },
      createBufferSource() {
        const node = {
          connect() {}, disconnect() {},
          start(time) { this.startTime = time; },
          stop() { this.stopped = true; },
        };
        nodes.push(node);
        return node;
      },
    },
  };
  return { context, state, nodes, elements };
}

test('WebRTC negotiates, displays data-channel transcripts, and closes its media', async () => {
  const { context, state, elements } = client();
  state.context = null;
  const microphone = { stop() { this.stopped = true; } };
  const speaker = { play: async () => {}, pause() { this.paused = true; }, remove() { this.removed = true; } };
  const createElement = context.document.createElement;
  context.document.createElement = tag => tag === 'audio' ? speaker : createElement();
  context.navigator = { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [microphone] }) } };
  context.location = { protocol: 'http:', host: 'localhost:8000' };
  context.MediaStream = class { constructor(tracks) { this.tracks = tracks; } };
  let channel;
  context.RTCPeerConnection = class {
    iceGatheringState = 'complete';
    createDataChannel() { channel = {}; return channel; }
    addTrack(track) { this.microphone = track; }
    async createOffer() { return { type: 'offer', sdp: 'offer-sdp' }; }
    async setLocalDescription(value) { this.localDescription = value; }
    async setRemoteDescription(value) { this.remoteDescription = value; }
    close() { this.closed = true; }
  };
  context.WebSocket = class {
    constructor(url) { this.url = url; }
    send(value) { this.sent = JSON.parse(value); }
    close() { this.closed = true; }
  };
  await context.startRtcCall(state);
  state.socket.onopen();
  assert.equal(state.socket.url, 'ws://localhost:8000/ws/offer');
  assert.equal(state.socket.sent.type, 'offer');
  await state.socket.onmessage({ data: JSON.stringify({ type: 'answer', sdp: 'answer-sdp' }) });
  assert.equal(state.pc.remoteDescription.sdp, 'answer-sdp');
  channel.onmessage({ data: JSON.stringify({ type: 'transcript', data: { turn_id: 1, text: 'Hello', is_final: true } }) });
  assert.equal(state.messages.get('you:1').body.textContent, 'Hello');
  assert.equal(elements.get('#emptyConversation').hidden, true);
  await context.endCall(state);
  assert.ok(state.pc.closed && state.socket.closed && microphone.stopped && speaker.removed);
});

test('PCM chunks are scheduled consecutively, not overlaid', () => {
  const { context, state, nodes } = client();
  context.playPcm(state, new ArrayBuffer(640));
  context.playPcm(state, new ArrayBuffer(640));
  assert.equal(nodes[0].startTime, 1.01);
  assert.equal(nodes[1].startTime, 1.03);
});

test('barge-in stops every scheduled source and clears partial PCM', () => {
  const { context, state, nodes } = client();
  context.playPcm(state, new ArrayBuffer(641));
  context.playPcm(state, new ArrayBuffer(640));
  context.stopPlayback(state);
  assert.ok(nodes.every(node => node.stopped));
  assert.equal(state.playing.size, 0);
  assert.equal(state.tail.length, 0);
  assert.equal(state.nextTime, 0);
});

test('PCM decoding preserves a sample split across network chunks', () => {
  const { context, state, nodes } = client();
  context.playPcm(state, new Uint8Array([0]).buffer);
  assert.equal(nodes.length, 0);
  context.playPcm(state, new Uint8Array([64, 0, 128]).buffer);
  assert.deepEqual([...nodes[0].buffer.data], [0.5, -1]);
});

test('AudioWorklet assembles 128-sample callbacks into 20 ms PCM16 frames', () => {
  const { context } = client();
  const sent = [];
  let Capture;
  const worklet = vm.createContext({
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage: data => sent.push(data) }; } },
    registerProcessor(name, cls) { Capture = cls; },
    ArrayBuffer, DataView, Math,
  });
  vm.runInContext(vm.runInContext('captureSource', context), worklet);
  const capture = new Capture();
  for (let i = 0; i < 5; i++) capture.process([[new Float32Array(128).fill(0.25)]]);
  assert.equal(sent.length, 2);
  for (const buffer of sent) {
    assert.equal(buffer.byteLength, 640);
    const view = new DataView(buffer);
    for (let offset = 0; offset < 640; offset += 2) assert.equal(view.getInt16(offset, true), 8192);
  }
});

test('user transcript revisions update one message and agent deltas stream separately', () => {
  const { context, state, elements } = client();
  const send = (type, data) => context.updateConversation(state, { type, data: { turn_id: 1, ...data } });
  send('transcript', { text: 'hel', is_final: false });
  send('transcript', { text: 'hello world', is_final: false });
  assert.equal(state.messages.get('you:1').body.textContent, 'hello world');
  assert.equal(state.messages.get('you:1').progress.textContent, 'Recognizing…');
  send('transcript', { text: 'Hello world.', is_final: true });
  send('agent_delta', { text: 'Hi ' });
  send('agent_delta', { text: 'there!' });
  assert.equal(state.messages.get('agent:1').body.textContent, 'Hi there!');
  send('agent_complete', { text: 'Hi there!', status: 'completed' });
  assert.equal(state.messages.get('agent:1').body.textContent, 'Hi there!');
  assert.equal(state.messages.get('agent:1').pending, false);
  assert.equal(state.messages.get('you:1').body.textContent, 'Hello world.');
  assert.equal(state.messages.get('you:1').progress.textContent, '');
  assert.equal(elements.get('#conversation').children.length, 2);
  assert.equal(elements.get('#emptyConversation').hidden, true);
});

test('interrupted replies retain their text and later turns create new messages', () => {
  const { context, state } = client();
  context.updateConversation(state, { type: 'agent_delta', data: { turn_id: 1, text: 'Here is a story…' } });
  context.updateConversation(state, { type: 'agent_complete', data: { turn_id: 1, text: 'Here is a story…', status: 'interrupted' } });
  context.updateConversation(state, { type: 'transcript', data: { turn_id: 2, text: 'stop', is_final: true } });
  context.updateConversation(state, { type: 'agent_delta', data: { turn_id: 2, text: 'Okay.' } });
  assert.equal(state.messages.get('agent:1').body.textContent, 'Here is a story…');
  assert.equal(state.messages.get('agent:1').progress.textContent, 'Interrupted');
  assert.equal(state.messages.get('you:2').body.textContent, 'stop');
  assert.equal(state.messages.get('agent:2').body.textContent, 'Okay.');
});

test('conversation renders literal HTML text and finishes pending messages on disconnect', () => {
  const { context, state } = client();
  const text = '<img src=x onerror="alert(1)">';
  context.updateConversation(state, { type: 'transcript', data: { turn_id: 1, text, is_final: false } });
  const entry = state.messages.get('you:1');
  assert.equal(entry.body.textContent, text);
  assert.equal(entry.body.children.length, 0);
  context.finishConversation(state);
  assert.equal(entry.body.textContent, text);
  assert.equal(entry.progress.textContent, 'Call ended');
  assert.equal(entry.pending, false);
});

test('a new speech turn ends the previous unfinished recognition', () => {
  const { context, state } = client();
  context.updateConversation(state, { type: 'transcript', data: { turn_id: 1, text: 'partial', is_final: false } });
  context.updateConversation(state, { type: 'speech_started', data: { turn_id: 2 } });
  assert.equal(state.messages.get('you:1').progress.textContent, 'Transcription interrupted');
  assert.equal(state.messages.get('you:1').pending, false);
});
