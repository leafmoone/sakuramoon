# M032 implementation report

The model stores blocks and condition biases under final-position names such as `slot_02`, while each depth registers only its active set. The dense reference concatenates text/style/image within each batch item and masks both query and key padding. Its output head receives only the image slice. T024 packed sequences are not flattened into one dense sample; the production path remains direct varlen FA4 under K001.
