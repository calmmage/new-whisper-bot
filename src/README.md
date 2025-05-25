Plan of the new workflow:

- Download audio to memory
    - if audio is too large:
        - offload to disk
        - cut in parts on disk
        - parts = [on-disk paths]
    - else:
        - parts = [audio - in-memory]
- process audio parts
    - use pydub to cut in-memory
    -
- sd
- asd
- asd