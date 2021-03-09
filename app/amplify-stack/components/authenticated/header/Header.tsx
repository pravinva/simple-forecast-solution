// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { useState, useEffect, useRef } from "react";
import { Container, Menu, MenuItem, styled } from "@material-ui/core";
import styles from "./Header.module.scss";
import { Button } from "../../common/button/Button";
import Auth, { CognitoUser } from "@aws-amplify/auth";
import Router from "next/router";

export const Header: React.SFC = ({ children }) => {
  const [user, setUser] = useState<any>();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const userMenuRef = useRef();

  useEffect(() => {
    loadUser();
  }, []);

  const loadUser = async () => {
    setUser(await Auth.currentUserInfo());
  };

  const handleSignOut = async () => {
    await Auth.signOut();
    Router.push("/");
  };

  return (
    <header className={styles.header}>
      <div className={styles.container}>
        <div className={styles.contents}>{children}</div>

        {user && (
          <div className={styles.userMenu}>
            <Button className={styles.avatar} innerRef={userMenuRef} onClick={() => setShowUserMenu(true)}>
              <img src="/images/avatar-sm.jpg" />
              {user?.attributes?.name || user?.attributes?.email}
            </Button>
            <Menu
              anchorEl={userMenuRef.current}
              anchorReference="anchorEl"
              anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
              keepMounted
              open={showUserMenu}
              getContentAnchorEl={null}
              onClose={() => setShowUserMenu(false)}
            >
              <MenuItem onClick={() => handleSignOut()}>Sign out</MenuItem>
            </Menu>
          </div>
        )}
      </div>
    </header>
  );
};
