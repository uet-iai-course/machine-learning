self.postMessage({ cmd: 'Ready', data: null });

const buildEmptySpeechMap = (size = 4096) => {
  const map = {};
  for (let i = 0; i <= size; i += 1) {
    map[String(i)] = {};
  }
  return map;
};

self.onmessage = (event) => {
  const message = event.data || {};
  const cmd = message.cmd;

  if (cmd === 'setup') {
    self.postMessage({
      cmd: 'Finished',
      data: { success: true, result: null }
    });
    return;
  }

  if (cmd === 'speech' || cmd === 'nextRules' || cmd === 'nextStyle') {
    self.postMessage({
      cmd: 'Finished',
      data: {
        success: true,
        result: JSON.stringify({
          options: {},
          translations: {},
          mactions: {},
          speech: buildEmptySpeechMap(),
          braille: {},
          label: '',
          ssml: '',
          braillelabel: ''
        })
      }
    });
    return;
  }

  self.postMessage({
    cmd: 'Finished',
    data: { success: true, result: null }
  });
};
