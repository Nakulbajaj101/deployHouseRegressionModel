#/bin/sh

rm -r dist/*

python -m pip install --upgrade build
python -m build

for pkg in $(ls dist)
do 
    echo "Pushing package $pkg to gemfury"
    curl -F package=@"dist/$pkg" $GEMFURY_PUSH_URL
done
