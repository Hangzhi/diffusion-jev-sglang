# Third-party components

- [SGLang](https://github.com/sgl-project/sglang), Apache-2.0. The project installs it separately and includes a small integration patch. The inference extension uses its public/internal engine interfaces.
- [LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini), model card lists Apache-2.0. Weights are downloaded from the pinned upstream revision and are not included in this repository.
- React and React DOM (MIT), Lucide icons (ISC), Vite (MIT), and their dependencies. Bundled JS preserves upstream license comments; exact package versions are in `web/package-lock.json`.
- [TypeSafe documentation](https://docs.typesafe.ai/api) is referenced to describe interoperability; no hosted Jev implementation, model, or branding is included.
- Kaggle Emojify is optional and is not redistributed. Review its dataset terms before downloading or publishing derived examples.

The bundled emoji examples and playground presets are original synthetic examples written for this project.

## JevBench

Comparison tools use datasets and scoring from [JevBench](https://github.com/fstandhartinger/jevbench), pinned to c6004e008ffba24aec091261ca1a5c02f7324702. Historical outcomes are attributed explicitly. The upstream MIT license follows:

MIT License

Copyright (c) 2026 Florian Standhartinger and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
