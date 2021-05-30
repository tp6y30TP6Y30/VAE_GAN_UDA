wget 'https://www.dropbox.com/s/812cmfc2d2tvqz1/decoder_99.ckpt?dl=1' -O ./Problem1/decoder_99.ckpt
wget 'https://www.dropbox.com/s/wc00cg9k0i4hu0k/encoder_99.ckpt?dl=1' -O ./Problem1/encoder_99.ckpt
python3 ./Problem1/main.py --mode test --pred_path $1 
