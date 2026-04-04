window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  },
  chtml: {
    adaptiveCSS: false
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  },
  startup: {
    ready() {
      const originalWarn = console.warn.bind(console);
      console.warn = (...args) => {
        const message = String(args[0] ?? '');
        if (
          message.includes('Component [mathjax-newcm]/chtml/dynamic/') ||
          message.includes('No version information available for component [mathjax-newcm]/chtml/dynamic/')
        ) {
          return;
        }
        originalWarn(...args);
      };
      if (window.MathJax?.loader) {
        window.MathJax.loader.checkVersion = () => {};
      }
      window.MathJax.startup.defaultReady();
    }
  },
  enableMenu: false,
  menuOptions: {
    settings: {
      enrich: false,
      speech: false,
      braille: false,
      collapsible: false,
      assistiveMml: false
    }
  }
};
