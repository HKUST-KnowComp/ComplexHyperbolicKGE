#!/bin/sh

make_sure_dir(){
    set -e
    name=$1
    [[ -z $name ]] && exit 1
    target_dir_name=$(dirname $name)
    [[ -d $target_dir_name ]] || mkdir -p $target_dir_name
    lock=$target_dir_name/'.mkdir.lock'
    exec 201>"$lock"
    until flock -n 201
    do
        sleep 0.1
    done
    if [[ -d $name ]] ; then
        i=0
        while [[ -d $name-$i ]] ; do
            i=$(( i+1 ))
        done
        name=$name-$i
    fi
    mkdir -p "$name"
    echo "$name"
    flock -u 201
}