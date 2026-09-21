# Third-party components

- [SGLang](https://github.com/sgl-project/sglang), Apache-2.0. The project installs it separately and includes a small integration patch. The inference extension uses its public/internal engine interfaces.
- [LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini), model card lists Apache-2.0. Weights are downloaded from the pinned upstream revision and are not included in this repository.
- React and React DOM (MIT), Lucide icons (ISC), Vite (MIT), and their dependencies. Bundled JS preserves upstream license comments; exact package versions are in `web/package-lock.json`.
- [TypeSafe documentation](https://docs.typesafe.ai/api) is referenced to describe interoperability; no hosted Jev implementation, model, or branding is included.
- Kaggle Emojify is optional and is not redistributed. Review its dataset terms before downloading or publishing derived examples.

The bundled emoji examples and playground presets are original synthetic examples written for this project.
