# D013 review scope

D013 is part of the Data package. Independent AI/model and Infra/performance conclusions remain pending until D010-D015 package review.

Review exact 17-shape generation, proportional 256/512/768/1024 scaling, post-EXIF dimensions, RGB conversion, no-upscale eligibility, nearest log-aspect selection, cover resize, 0.80 retention rejection, and deterministic uniform crop offsets.

Also review exact D010 aggregate binding for the streaming metadata scan,
assigned/rejection/per-bucket accounting, exactly 100,000 post-EXIF observations, the
inclusive 0.1% mismatch boundary, diagnostic failure at 101 mismatches, canonical JSON,
unique temporary publication, fsync, atomic no-clobber linking, and cleanup.

Do not attribute an executed production metadata scan, real 100k decode report,
production retention distribution, CPU throughput, VAE reconstruction quality, or D015
stage/pass/sample seed derivation to this task.
