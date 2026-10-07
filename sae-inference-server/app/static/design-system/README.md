# Demo brand assets

The demo follows the supplied `Video Asset Checklist.html` and
`design-system/colors_and_type.css`: Signifier Light headlines, Roboto UI,
the Dataiku core palette, soft cards, and subtle ease-out transitions.

The original black Dataiku lockup and white logomark are in `../logos/`.
They are used without recoloring or effects. Fonts and artwork are served
locally through `/demo/assets/` and copied into the sidecar image.

Source bundle: `DemosVideos/Video Asset Checklist.html`, `design-system/`,
and `dataiku-video-kit/logos/`. The referenced Signifier files were supplied
in `Downloads/Dataiku Design System/fonts/`. The design tokens are retained;
the Google Fonts import is disabled so isolated deployments can use the
declared system monospace fallback without external requests.
