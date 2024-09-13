#/bin/bash

mkdir pretrained
mkdir pretrained/vimeo
mkdir pretrained/vizdrone
mkdir pretrained/mami

wget https://arizona.box.com/shared/static/l0dclotorf4fgoiruhztqx26btgb3wbz.pkl -P pretrained/vimeo/ -O pretrained/vimeo/HSTR_ifnet_62.pkl
wget https://arizona.box.com/shared/static/96lcnmtn1zrha2y7el7940gqg3r191l0.pkl -P pretrained/vimeo/ -O pretrained/vimeo/HSTR_unet_62.pkl
wget https://arizona.box.com/shared/static/47yzmk4k869hqszoyegdnc6vi4jccc2r.pkl -P pretrained/vimeo/ -O pretrained/vimeo/HSTR_attention_62.pkl
wget https://arizona.box.com/shared/static/pkpgzlr75jdh1s3ydpr2mfwleobtka4k.pkl -P pretrained/vimeo/ -O pretrained/vimeo/HSTR_contextnet_62.pkl


wget https://arizona.box.com/shared/static/st33vv9jwy0q4idmw7gklwpai7ficz7a.pkl -P pretrained/vizdrone/ -O pretrained/vizdrone/HSTR_ifnet_39.pkl
wget https://arizona.box.com/shared/static/n5f48m4jwsrwrobxb8agkikj7fuvz1n3.pkl -P pretrained/vizdrone/ -O pretrained/vizdrone/HSTR_unet_39.pkl
wget https://arizona.box.com/shared/static/e7g4i6brrfwneofdkquhk8ldbct1pn6i.pkl -P pretrained/vizdrone/ -O pretrained/vizdrone/HSTR_attention_39.pkl
wget https://arizona.box.com/shared/static/3hoypnyapsj6i9v6pbxwfq3dftozg1a8.pkl -P pretrained/vizdrone/ -O pretrained/vizdrone/HSTR_contextnet_39.pkl


wget https://arizona.box.com/shared/static/w7lv57tpgnq3vfnx70qfltn7x9q9ffuv.pkl -P pretrained/mami/ -O pretrained/mami/HSTR_ifnet_68.pkl
wget https://arizona.box.com/shared/static/52flk4jpn1smup3czvlc960zee6brhdb.pkl -P pretrained/mami/ -O pretrained/mami/HSTR_unet_68.pkl
wget https://arizona.box.com/shared/static/s4pl3mye7ppsl1y4c1vyq04alr6ztntc.pkl -P pretrained/mami/ -O pretrained/mami/HSTR_attention_68.pkl
wget https://arizona.box.com/shared/static/0ndhegx501lsj1hgowm8oksa5drwi65l.pkl -P pretrained/mami/ -O pretrained/mami/HSTR_contextnet_68.pkl