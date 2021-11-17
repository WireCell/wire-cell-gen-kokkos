function setup_supported_version {
  if (ups active | grep -q $1)
  then
      return
  fi

  local package_entry=$(ups list -aK+ $1 | sort -V)
  if [ -z "$package_entry" ]
  then
      echo "$1 is not available -- ignoring"
      return 1
  fi

  if [ $(echo "$package_entry" | wc -l) != "1" ]
  then
      package_entry=$(echo "$package_entry" | grep -v "c7\|debug" | tail -1)
      echo "Multiple versions for package $1 found; picking most recent:"
      echo "  $package_entry"
  fi

  ups_list_pattern='"(.*)" "(.*)" ".*" "(.*)" ".*"'
  if [[ ! $package_entry =~ $ups_list_pattern ]]
  then
      echo "Malformed ups list entry for $1 -- fatal error"
      return 2
  fi

  local product=${BASH_REMATCH[1]}
  local version=${BASH_REMATCH[2]}
  local quals=${BASH_REMATCH[3]}
  [ -n "$quals" ] && quals="-q $quals" || quals=""
  setup $product $version $quals
}

type ups >& /dev/null || source /products/setup
setup_supported_version larwirecell
setup_supported_version cmake
setup_supported_version gdb
setup_supported_version valgrind

WIRECELL_PATH=$(realpath $WIRECELL_FQ_DIR/wirecell-*/cfg):$WIRECELL_PATH
