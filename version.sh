#!/bin/bash
export LC_ALL=C
# 生成 svn 状态信息
svn_info_txt=`svn info`; ret_info=$?
unset svn_info_txt

if [ $ret_info -eq 0 ]
then {
    #revision=`git describe`;
    svn_version=`svn info | grep Revision | grep -o '[0-9]\+'`;
    svn log -r$svn_version > svn_log; ret_log=$?
    if [ $ret_log -eq 0 ]
    then {
        svn_log=`sed '1d;2d;3d;$d' svn_log | tr '\n' ' '`;
    } fi
    rm svn_log -f
} fi

#svn_diff=`svn diff`; ret_diff=$?
#if [ $ret_diff -eq 0 ]
#then {
#    svn_log=`sed '1d;2d;3d;$d' svn_log`;
#} fi
sf_author=$LOGNAME
sf_date=`date -R`

# 处理生成的状态信息
echo "#define SE_COMPILE_INFO \"super_enc_v3_test author: $sf_author time: $sf_date version: r${svn_version:-0}\"" > ./svn_info.h
